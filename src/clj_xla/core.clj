(ns clj-xla.core
  "High-level JIT compilation and execution API for clj-xla."
  (:require [clj-xla.compile :as compile]
            [clj-xla.pjrt :as pjrt]))

(defonce ^:private default-context (atom nil))

(defn init-cpu!
  "Initializes the CPU PJRT runtime client. Uses bin/libpjrt_cpu.so by default."
  ([]
   (init-cpu! "bin/libpjrt_cpu.so"))
  ([lib-path]
   (let [api (pjrt/load-plugin! lib-path)
         client (pjrt/create-client api)
         pname (pjrt/platform-name api client)
         ctx {:api api :client client :platform pname}]
     (reset! default-context ctx)
     (println (str "clj-xla initialized PJRT Backend: [" pname "]"))
     ctx)))

(defn get-context
  "Returns the default initialized runtime context, initializing CPU runtime if needed."
  []
  (or @default-context (init-cpu!)))

(defn compile-graph
  "Compiles an EDN SSA graph against the active runtime context."
  ([graph]
   (compile-graph (get-context) graph))
  ([{:keys [api client]} graph]
   (compile/compile-graph api client graph)))
