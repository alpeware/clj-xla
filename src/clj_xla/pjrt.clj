(ns clj-xla.pjrt
  "Project Panama FFM native bindings to OpenXLA PJRT C API (pjrt_c_api.h)."
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
(def OFFSET_CLIENT_CREATE "Offset of PJRT_Client_Create in PJRT_Api." 120)
(def OFFSET_CLIENT_DESTROY "Offset of PJRT_Client_Destroy in PJRT_Api." 128)
(def OFFSET_CLIENT_PLATFORM_NAME "Offset of PJRT_Client_PlatformName in PJRT_Api." 136)
(def OFFSET_CLIENT_PLATFORM_VERSION "Offset of PJRT_Client_PlatformVersion in PJRT_Api." 152)
(def OFFSET_CLIENT_COMPILE "Offset of PJRT_Client_Compile in PJRT_Api." 200)
(def OFFSET_CLIENT_BUFFER_FROM_HOST_BUFFER "Offset of PJRT_Client_BufferFromHostBuffer in PJRT_Api." 216)

(def NO_OPTIONS "Empty Linker$Option array." (into-array Linker$Option []))
(def NO_LAYOUTS "Empty MemoryLayout array." (into-array MemoryLayout []))

;; Minimal binary Protobuf serialization for CompileOptionsProto:
;; CompileOptionsProto { ExecutableBuildOptionsProto executable_build_options (tag 0x1a) { num_replicas = 1 (tag 0x20), num_partitions = 1 (tag 0x28) } }
(def DEFAULT_COMPILE_OPTIONS_BYTES
  "Default serialized CompileOptionsProto bytes configuring num_replicas=1 and num_partitions=1."
  (byte-array [(byte 0x1a) (byte 4) (byte 0x20) (byte 1) (byte 0x28) (byte 1)]))

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
    (let [{:keys [api-ptr linker arena]} api-ctx
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

(defn load-plugin!
  "Loads the PJRT shared object from `lib-path` and initializes the PJRT plugin."
  [lib-path]
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
  (let [{:keys [api-ptr linker arena]} api-ctx
        create-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_CREATE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        create-args (.allocate ^Arena arena (long 88))]
    (.set ^MemorySegment create-args ValueLayout/JAVA_LONG (long 0) (long 88))
    (let [err (.invokeWithArguments ^MethodHandle create-handle [create-args])]
      (check-error! api-ctx err)
      (.get ^MemorySegment create-args ValueLayout/ADDRESS (long 64)))))

(defn platform-name
  "Queries the platform name string from `client` (e.g. 'cpu', 'cuda')."
  [api-ctx client]
  (let [{:keys [api-ptr linker arena]} api-ctx
        pname-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_PLATFORM_NAME ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        pname-args (.allocate ^Arena arena (long 40))]
    (.set ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 0) (long 40))
    (.set ^MemorySegment pname-args ValueLayout/ADDRESS (long 16) client)
    (let [err (.invokeWithArguments ^MethodHandle pname-handle [pname-args])]
      (check-error! api-ctx err)
      (let [str-ptr (.get ^MemorySegment pname-args ValueLayout/ADDRESS (long 24))
            str-len (.get ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 32))
            str-bytes (.toArray (.reinterpret ^MemorySegment str-ptr str-len) ValueLayout/JAVA_BYTE)]
        (String. ^bytes str-bytes "UTF-8")))))

(defn compile-mlir
  "Compiles `mlir-text` into a PjRtLoadedExecutable native handle."
  [api-ctx client mlir-text]
  (let [{:keys [api-ptr linker arena]} api-ctx
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
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 16) client)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 24) program)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 32) opts-seg)
      (.set ^MemorySegment compile-args ValueLayout/JAVA_LONG (long 40) (long (alength DEFAULT_COMPILE_OPTIONS_BYTES)))

      ;; PJRT_Client_Compile at offset 200
      (let [compile-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_COMPILE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            err (.invokeWithArguments ^MethodHandle compile-handle [compile-args])]
        (check-error! api-ctx err)
        (.get ^MemorySegment compile-args ValueLayout/ADDRESS (long 48))))))
