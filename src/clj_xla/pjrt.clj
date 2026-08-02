(ns clj-xla.pjrt
  "Project Panama FFM native bindings to OpenXLA PJRT C API (pjrt_c_api.h)."
  (:require [clojure.java.io :as io])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.file Path]))

;; PJRT_Api struct field offsets (64-bit architecture)
(def OFFSET_STRUCT_SIZE "Offset of struct_size in PJRT_Api." 0)
(def OFFSET_EXTENSION_START "Offset of extension_start in PJRT_Api." 8)
(def OFFSET_API_VERSION "Offset of pjrt_api_version in PJRT_Api." 16)
(def OFFSET_ERROR_DESTROY "Offset of PJRT_Error_Destroy in PJRT_Api." 40)
(def OFFSET_ERROR_MESSAGE "Offset of PJRT_Error_Message in PJRT_Api." 48)
(def OFFSET_ERROR_GET_CODE "Offset of PJRT_Error_GetCode in PJRT_Api." 56)
(def OFFSET_PLUGIN_INITIALIZE "Offset of PJRT_Plugin_Initialize in PJRT_Api." 64)
(def OFFSET_PLUGIN_ATTRIBUTES "Offset of PJRT_Plugin_Attributes in PJRT_Api." 72)
(def OFFSET_EVENT_DESTROY "Offset of PJRT_Event_Destroy in PJRT_Api." 80)
(def OFFSET_EVENT_AWAIT "Offset of PJRT_Event_Await in PJRT_Api." 104)
(def OFFSET_CLIENT_CREATE "Offset of PJRT_Client_Create in PJRT_Api." 120)
(def OFFSET_CLIENT_DESTROY "Offset of PJRT_Client_Destroy in PJRT_Api." 128)
(def OFFSET_CLIENT_PLATFORM_NAME "Offset of PJRT_Client_PlatformName in PJRT_Api." 136)
(def OFFSET_CLIENT_PLATFORM_VERSION "Offset of PJRT_Client_PlatformVersion in PJRT_Api." 152)
(def OFFSET_CLIENT_ADDRESSABLE_DEVICES "Offset of PJRT_Client_AddressableDevices in PJRT_Api." 168)
(def OFFSET_CLIENT_COMPILE "Offset of PJRT_Client_Compile in PJRT_Api." 200)
(def OFFSET_CLIENT_BUFFER_FROM_HOST_BUFFER "Offset of PJRT_Client_BufferFromHostBuffer in PJRT_Api." 216)
(def OFFSET_LOADED_EXECUTABLE_EXECUTE "Offset of PJRT_LoadedExecutable_Execute in PJRT_Api." 480)
(def OFFSET_BUFFER_DESTROY "Offset of PJRT_Buffer_Destroy in PJRT_Api." 504)
(def OFFSET_BUFFER_TO_HOST_BUFFER "Offset of PJRT_Buffer_ToHostBuffer in PJRT_Api." 600)

(def NO_OPTIONS "Empty Linker$Option array." (into-array Linker$Option []))
(def NO_LAYOUTS "Empty MemoryLayout array." (into-array MemoryLayout []))

;; Minimal binary Protobuf serialization for CompileOptionsProto:
;; CompileOptionsProto { ExecutableBuildOptionsProto executable_build_options (tag 0x1a) { num_replicas = 1 (tag 0x20), num_partitions = 1 (tag 0x28) } }
(def DEFAULT_COMPILE_OPTIONS_BYTES
  "Default serialized CompileOptionsProto bytes configuring num_replicas=1 and num_partitions=1."
  (byte-array [(byte 0x1a) (byte 4) (byte 0x20) (byte 1) (byte 0x28) (byte 1)]))

(defn- extract-ctx [ctx]
  (cond
    (nil? ctx) {}
    (:api-ptr ctx) ctx
    (:api ctx) (:api ctx)
    :else ctx))

(defn- extract-client [ctx client]
  (or client (:client ctx)))

(defn- downcall-ptr
  "Creates a downcall MethodHandle for function pointer at `offset` inside `api-ptr`."
  [^Linker linker ^MemorySegment api-ptr offset return-layout arg-layouts]
  (let [^MemorySegment fn-ptr (.get ^MemorySegment api-ptr ValueLayout/ADDRESS (long offset))
        ^FunctionDescriptor fd (if (some? return-layout)
                                 (FunctionDescriptor/of return-layout (into-array MemoryLayout arg-layouts))
                                 (FunctionDescriptor/ofVoid (into-array MemoryLayout arg-layouts)))]
    (.downcallHandle linker fn-ptr fd NO_OPTIONS)))

(defn check-error!
  "Checks if `err-ptr` is non-null. If so, extracts error message, frees error, and throws Exception."
  [api-ctx err-ptr]
  (when (and (some? err-ptr) (not= MemorySegment/NULL err-ptr))
    (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
          msg-handle (downcall-ptr linker api-ptr OFFSET_ERROR_MESSAGE nil [ValueLayout/ADDRESS])
          msg-args (.allocate ^Arena arena (long 40))]
      (.set ^MemorySegment msg-args ValueLayout/JAVA_LONG (long 0) (long 40))
      (.set ^MemorySegment msg-args ValueLayout/ADDRESS (long 16) err-ptr)
      (.invokeWithArguments ^MethodHandle msg-handle [msg-args])
      (let [str-ptr (.get ^MemorySegment msg-args ValueLayout/ADDRESS (long 24))
            str-len (.get ^MemorySegment msg-args ValueLayout/JAVA_LONG (long 32))
            str-bytes (.toArray (.reinterpret ^MemorySegment str-ptr str-len) ValueLayout/JAVA_BYTE)
            err-msg (String. ^bytes str-bytes "UTF-8")
            destroy-handle (downcall-ptr linker api-ptr OFFSET_ERROR_DESTROY nil [ValueLayout/ADDRESS])
            destroy-args (.allocate ^Arena arena (long 24))]
        (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
        (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) err-ptr)
        (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])
        (throw (ex-info (str "PJRT Error: " err-msg) {:error-msg err-msg}))))))

(defn- preload-libpython!
  "Pre-loads system libpython library with RTLD_GLOBAL (0x102) if present, resolving Python C API symbols for PyPI PJRT plugins."
  []
  (let [paths ["/usr/lib64/libpython3.12.so"
               "/usr/lib64/libpython3.13.so"
               "/usr/lib64/libpython3.14.so"
               "/usr/lib64/libpython3.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.12.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.11.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.10.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.so"]
        existing (first (filter #(and (.exists (io/file %)) (not (.isDirectory (io/file %)))) paths))]
    (when existing
      (try
        (let [linker (Linker/nativeLinker)
              stdlib (.defaultLookup linker)
              ^MemorySegment dlopen-seg (.get (.find stdlib "dlopen"))
              ^FunctionDescriptor desc (FunctionDescriptor/of ValueLayout/ADDRESS (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
              dlopen-fn (.downcallHandle linker dlopen-seg desc (make-array Linker$Option 0))
              arena (Arena/global)
              py-path (.allocateFrom arena ^String existing)]
          (.invokeWithArguments dlopen-fn [py-path (Integer/valueOf 258)]))
        (catch Exception _ nil)))))

(defn load-plugin!
  "Loads the PJRT shared object from `lib-path` and initializes the PJRT plugin."
  [lib-path]
  (preload-libpython!)
  (let [arena (Arena/global)
        abs-path (.toAbsolutePath (Path/of lib-path (into-array String [])))
        lookup (SymbolLookup/libraryLookup abs-path arena)
        ^MemorySegment get-api (.orElseThrow (.find lookup "GetPjrtApi"))
        linker (Linker/nativeLinker)
        ^FunctionDescriptor fd (FunctionDescriptor/of ValueLayout/ADDRESS NO_LAYOUTS)
        ^MethodHandle handle (.downcallHandle linker get-api fd NO_OPTIONS)
        raw-api (.invokeWithArguments handle [])
        ^MemorySegment api-ptr (.reinterpret ^MemorySegment raw-api 984)
        api-ctx {:api-ptr api-ptr :linker linker :arena arena}]

    ;; Call PJRT_Plugin_Initialize
    (let [init-handle (downcall-ptr linker api-ptr OFFSET_PLUGIN_INITIALIZE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          init-args (.allocate arena (long 16))]
      (.set ^MemorySegment init-args ValueLayout/JAVA_LONG (long 0) (long 16))
      (let [err (.invokeWithArguments ^MethodHandle init-handle [init-args])]
        (check-error! api-ctx err)))
    api-ctx))

(defn create-client
  "Creates a PJRT_Client for the loaded plugin."
  [api-ctx]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        create-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_CREATE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        create-args (.allocate ^Arena arena (long 88))]
    (.set ^MemorySegment create-args ValueLayout/JAVA_LONG (long 0) (long 88))
    (let [err (.invokeWithArguments ^MethodHandle create-handle [create-args])]
      (check-error! api-ctx err)
      (.get ^MemorySegment create-args ValueLayout/ADDRESS (long 64)))))

(defn platform-name
  "Queries the platform name string from `client` (e.g. 'cpu', 'cuda')."
  [api-ctx client]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        pname-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_PLATFORM_NAME ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        pname-args (.allocate ^Arena arena (long 40))]
    (.set ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 0) (long 40))
    (.set ^MemorySegment pname-args ValueLayout/ADDRESS (long 16) cli)
    (let [err (.invokeWithArguments ^MethodHandle pname-handle [pname-args])]
      (check-error! api-ctx err)
      (let [str-ptr (.get ^MemorySegment pname-args ValueLayout/ADDRESS (long 24))
            str-len (.get ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 32))
            str-bytes (.toArray (.reinterpret ^MemorySegment str-ptr str-len) ValueLayout/JAVA_BYTE)]
        (String. ^bytes str-bytes "UTF-8")))))

(defn addressable-devices
  "Returns array of addressable PJRT_Device handles for `client`."
  [api-ctx client]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        args (.allocate ^Arena arena (long 40))]
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 40))
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) cli)
    (let [handle (downcall-ptr linker api-ptr OFFSET_CLIENT_ADDRESSABLE_DEVICES ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          err (.invokeWithArguments ^MethodHandle handle [args])]
      (check-error! api-ctx err)
      (let [devs-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 24))
            num-devs (.get ^MemorySegment args ValueLayout/JAVA_LONG (long 32))]
        (mapv (fn [i]
                (.getAtIndex (.reinterpret ^MemorySegment devs-ptr (* num-devs 8)) ValueLayout/ADDRESS (long i)))
              (range num-devs))))))

(defn compile-mlir
  "Compiles `mlir-text` into a PjRtLoadedExecutable native handle."
  [api-ctx client mlir-text]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        code-bytes (.getBytes ^String mlir-text "UTF-8")
        code-seg (.allocateFrom ^Arena arena (String. ^bytes code-bytes "UTF-8"))
        format-seg (.allocateFrom ^Arena arena "mlir")
        opts-seg (.allocateFrom ^Arena arena ValueLayout/JAVA_BYTE DEFAULT_COMPILE_OPTIONS_BYTES)

        ;; PJRT_Program (size 48)
        program (.allocate ^Arena arena (long 48))]
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 0) (long 48))
    (.set ^MemorySegment program ValueLayout/ADDRESS (long 16) code-seg)
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 24) (long (alength code-bytes)))
    (.set ^MemorySegment program ValueLayout/ADDRESS (long 32) format-seg)
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 40) (long 4))

    ;; PJRT_Client_Compile_Args (size 56)
    (let [compile-args (.allocate ^Arena arena (long 56))]
      (.set ^MemorySegment compile-args ValueLayout/JAVA_LONG (long 0) (long 56))
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 16) cli)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 24) program)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 32) opts-seg)
      (.set ^MemorySegment compile-args ValueLayout/JAVA_LONG (long 40) (long (alength DEFAULT_COMPILE_OPTIONS_BYTES)))

      ;; PJRT_Client_Compile at offset 200
      (let [compile-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_COMPILE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            err (.invokeWithArguments ^MethodHandle compile-handle [compile-args])]
        (check-error! api-ctx err)
        (.get ^MemorySegment compile-args ValueLayout/ADDRESS (long 48))))))

(defn buffer-from-host-buffer
  "Transfers a host memory buffer (float-array or int-array) to a device PJRT_Buffer."
  [api-ctx client host-data shape dtype-enum]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        num-dims (count shape)
        devs (addressable-devices api-ctx cli)
        dev (first devs)]
    (with-open [arena (Arena/ofConfined)]
      (let [dims-seg (.allocate arena ValueLayout/JAVA_LONG (long num-dims))]
        (dotimes [i num-dims]
          (.setAtIndex ^MemorySegment dims-seg ValueLayout/JAVA_LONG (long i) (long (nth shape i))))
        (let [data-seg (cond
                         (instance? MemorySegment host-data)
                         (if (.isNative ^MemorySegment host-data)
                           host-data
                           (.allocateFrom arena ValueLayout/JAVA_BYTE ^MemorySegment host-data))

                         (= (int dtype-enum) 4)
                         (let [^ints ia (if (instance? (Class/forName "[I") host-data)
                                          ^ints host-data
                                          (int-array (map int (flatten host-data))))]
                           (.allocateFrom arena ValueLayout/JAVA_INT ia))

                         :else
                         (let [^floats fa (if (instance? (Class/forName "[F") host-data)
                                            ^floats host-data
                                            (float-array (map float (flatten host-data))))
                               num-floats (alength fa)
                               byte-size (* (long num-floats) 4)
                               seg (.allocate arena byte-size (long 64))]
                           (MemorySegment/copy fa 0 seg ValueLayout/JAVA_FLOAT (long 0) num-floats)
                           seg))
              args (.allocate arena (long 120))]
          (.fill args (byte 0))
          (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 120))
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) cli)
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) data-seg)
          (.set ^MemorySegment args ValueLayout/JAVA_INT (long 32) (int dtype-enum))
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 40) dims-seg)
          (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 48) (long num-dims))
          (when dev
            (.set ^MemorySegment args ValueLayout/ADDRESS (long 80) dev))
          (let [handle (downcall-ptr linker api-ptr OFFSET_CLIENT_BUFFER_FROM_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                err (.invokeWithArguments ^MethodHandle handle [args])]
            (check-error! api-ctx err)
            (.get ^MemorySegment args ValueLayout/ADDRESS (long 112))))))))

(defn destroy-buffer!
  "Frees native device PJRT_Buffer `buffer-handle`."
  [api-ctx buffer-handle]
  (when (and (some? buffer-handle) (not= MemorySegment/NULL buffer-handle))
    (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
          arena (Arena/ofAuto)
          args (.allocate arena (long 24))]
      (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 24))
      (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buffer-handle)
      (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            err (.invokeWithArguments ^MethodHandle handle [args])]
        (check-error! api-ctx err)))))

(defn execute-executable
  "Executes compiled PjRtLoadedExecutable native handle `exec-handle` with device `PjRtBuffer` handles `input-buffers`."
  [api-ctx exec-handle input-buffers]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        cli (extract-client api-ctx nil)
        devs (when cli (addressable-devices api-ctx cli))
        dev (first devs)
        num-args (count input-buffers)
        arena (Arena/ofAuto)
        arg-ptrs (.allocate arena ValueLayout/ADDRESS (long num-args))]
    (dotimes [i num-args]
      (.setAtIndex ^MemorySegment arg-ptrs ValueLayout/ADDRESS (long i) (nth input-buffers i)))
    (let [arg-lists (.allocate arena ValueLayout/ADDRESS (long 1))
          _ (.setAtIndex ^MemorySegment arg-lists ValueLayout/ADDRESS (long 0) arg-ptrs)
          out-ptrs (.allocate arena ValueLayout/ADDRESS (long 1))
          out-lists (.allocate arena ValueLayout/ADDRESS (long 1))
          _ (.setAtIndex ^MemorySegment out-lists ValueLayout/ADDRESS (long 0) out-ptrs)
          opts (.allocate arena (long 112))
          _ (.set ^MemorySegment opts ValueLayout/JAVA_LONG (long 0) (long 112))
          args (.allocate arena (long 80))]
      (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 80))
      (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) exec-handle)
      (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) opts)
      (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) arg-lists)
      (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long 1))
      (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 48) (long num-args))
      (.set ^MemorySegment args ValueLayout/ADDRESS (long 56) out-lists)
      (when dev
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 72) dev))
      (let [handle (downcall-ptr linker api-ptr OFFSET_LOADED_EXECUTABLE_EXECUTE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            err (.invokeWithArguments ^MethodHandle handle [args])]
        (check-error! api-ctx err)
        (.get ^MemorySegment out-ptrs ValueLayout/ADDRESS (long 0))))))

(defn buffer-to-host-buffer
  "Copies device PJRT_Buffer `buffer-handle` to host float array, awaiting asynchronous completion."
  [api-ctx buffer-handle num-floats]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        byte-size (* num-floats 4)
        arena (Arena/ofAuto)
        dst-seg (.allocate arena ValueLayout/JAVA_FLOAT (long num-floats))
        args (.allocate arena (long 56))]
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 56))
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buffer-handle)
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) dst-seg)
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long byte-size))
    (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_TO_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          err (.invokeWithArguments ^MethodHandle handle [args])]
      (check-error! api-ctx err)
      (let [event-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 48))]
        (when (and (some? event-ptr) (not= MemorySegment/NULL event-ptr))
          (let [await-args (.allocate arena (long 24))]
            (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
            (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) event-ptr)
            (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                  await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
              (check-error! api-ctx await-err)))
          (let [destroy-args (.allocate arena (long 24))]
            (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
            (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) event-ptr)
            (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                  _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])]))))
      (.toArray dst-seg ValueLayout/JAVA_FLOAT))))
