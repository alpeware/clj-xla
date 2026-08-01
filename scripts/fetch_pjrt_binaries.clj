(ns scripts.fetch-pjrt-binaries
  "Automated binary fetcher for precompiled PJRT shared objects from PyPI wheels."
  (:require [clojure.java.io :as io])
  (:import [java.util.zip ZipInputStream]))

(def SOURCES
  "Map of target backend to PyPI wheel download URL and extraction parameters."
  {:cpu
   {:url "https://files.pythonhosted.org/packages/5d/ca/5487881b30d07396c168749ead36f57dbc4538cafdc535943afc81b3e5a2/xla_cpu_pjrt-0.0.1-py3-none-manylinux_2_27_x86_64.whl"
    :so-name "libpjrt_cpu.so"
    :entry-pattern #"xla_cpu_pjrt\.so$"}

   :cuda12
   {:url "https://files.pythonhosted.org/packages/5f/78/a3d9ceda0793f4fb43daa292af7b801932611a1aed442636ddfc93d58c7a/jax_cuda12_pjrt-0.10.0-py3-none-manylinux_2_27_x86_64.whl"
    :so-name "libpjrt_cuda.so"
    :entry-pattern #"xla_cuda_plugin\.so$"}

   :sycl
   {:url "https://files.pythonhosted.org/packages/6a/45/95a2a7b82f6257b73c1e4db02804df1592fbee0e0469103a016f1c45e654/intel_extension_for_openxla-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    :so-name "libpjrt_sycl.so"
    :entry-pattern #"(xla_sycl_plugin|pjrt_sycl_plugin|pjrt_plugin_xpu)\.so$"}

   :rocm
   {:url "https://files.pythonhosted.org/packages/0c/84/bd62856b8ff221a5e8af4da41fe6f80667ad8ce72140e60a9a73c945144f/jax_rocm60_pjrt-0.4.35-py3-none-manylinux_2_28_x86_64.whl"
    :so-name "libpjrt_rocm.so"
    :entry-pattern #"(xla_rocm_plugin|pjrt_rocm_plugin)\.so$"}})

(defn unpack-so-from-wheel
  "Streams the zip archive from `url` and extracts the `.so` matching `entry-pattern` to `out-file`."
  [url entry-pattern out-file]
  (with-open [in (io/input-stream url)
              zip (ZipInputStream. in)]
    (loop []
      (when-let [entry (.getNextEntry zip)]
        (if (re-find entry-pattern (.getName entry))
          (do
            (println (str "Extracting " (.getName entry) " -> " (.getAbsolutePath out-file)))
            (io/copy zip out-file))
          (recur))))))

(defn fetch-binary
  "Fetches and unpacks the native shared object for `target` (:cpu, :cuda12, :sycl, :rocm) into bin/."
  [target]
  (let [{:keys [url so-name entry-pattern]} (get SOURCES target)
        _ (assert (some? url) (str "Unknown target: " target ". Available: " (keys SOURCES)))
        out-dir (io/file "bin")
        out-file (io/file out-dir so-name)]
    (.mkdirs out-dir)
    (println (str "Fetching " target " PJRT plugin from " url "..."))
    (unpack-so-from-wheel url entry-pattern out-file)
    (println (str "Successfully installed " (.getAbsolutePath out-file)))))

(defn -main
  "CLI entrypoint for binary fetcher script.
   Usage: clj -M:fetch-pjrt [cpu|cuda12|sycl|rocm|all]"
  [& args]
  (let [target-str (or (first args) "cpu")]
    (if (= target-str "all")
      (doseq [t (keys SOURCES)]
        (fetch-binary t))
      (fetch-binary (keyword target-str)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
