(ns scripts.download-hf
  "Download Hugging Face model weights using pure Clojure and JDK HttpClient."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.net URI]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest HttpResponse$BodyHandlers]
           [java.nio.file Files Paths]))

(defn load-hf-token
  "Loads HF API token from .env map or HF_TOKEN environment variable."
  []
  (let [env-file (io/file ".env")]
    (if (.exists env-file)
      (let [data (edn/read-string (slurp env-file))]
        (or (:hf-token data) (:token data) (:HF_TOKEN data)))
      (System/getenv "HF_TOKEN"))))

(defn- http-client []
  (.. (HttpClient/newBuilder)
      (followRedirects HttpClient$Redirect/NORMAL)
      build))

(defn parse-manifest-rpaths
  "Extracts file rpaths or rfilenames from Hugging Face model API body."
  [body]
  (mapv second (re-seq #"\"r(?:filename|path)\":\"([^\"]+)\"" body)))

(defn filter-relevant-files
  "Filters repository files to target model weights, tokenizers, and configs, prioritizing metadata and tokenizer files."
  [files]
  (let [matched (filterv #(re-find #"(safetensors|json|txt|model)$" %) files)]
    (sort-by (fn [f] (if (str/includes? f "safetensors") 1 0)) matched)))

(defn- fetch-repo-files [client repo token]
  (let [url (str "https://huggingface.co/api/models/" repo)
        req (cond-> (HttpRequest/newBuilder)
              true (.uri (URI/create url))
              token (.header "Authorization" (str "Bearer " token))
              true (.GET)
              true (.build))
        resp (.send client req (HttpResponse$BodyHandlers/ofString))
        body (.body resp)]
    (parse-manifest-rpaths body)))

(defn- download-file [client repo filename target-dir token]
  (let [url (str "https://huggingface.co/" repo "/resolve/main/" filename)
        out-path (Paths/get target-dir (into-array String [(last (str/split filename #"/"))]))
        parent-file (.toFile (.getParent out-path))
        _ (when parent-file (.mkdirs parent-file))
        existing-size (if (Files/exists out-path (into-array java.nio.file.LinkOption []))
                        (Files/size out-path)
                        0)
        req (cond-> (HttpRequest/newBuilder)
              true (.uri (URI/create url))
              token (.header "Authorization" (str "Bearer " token))
              (pos? existing-size) (.header "Range" (str "bytes=" existing-size "-"))
              true (.GET)
              true (.build))]
    (println (format "  --> Downloading %s (resume from %d MB)..." filename (quot existing-size (* 1024 1024))))
    (let [resp (.send client req (HttpResponse$BodyHandlers/ofInputStream))
          in-stream (.body resp)
          status (.statusCode resp)]
      (if (or (= status 200) (= status 206))
        (let [open-options (if (= status 206)
                             [java.nio.file.StandardOpenOption/APPEND java.nio.file.StandardOpenOption/WRITE]
                             [java.nio.file.StandardOpenOption/CREATE java.nio.file.StandardOpenOption/TRUNCATE_EXISTING java.nio.file.StandardOpenOption/WRITE])]
          (with-open [out-stream (Files/newOutputStream out-path (into-array java.nio.file.OpenOption open-options))]
            (io/copy in-stream out-stream))
          (println (format "      Done: %s (%d MB)"
                           (.getFileName out-path)
                           (quot (Files/size out-path) (* 1024 1024)))))
        (println (format "      Error downloading %s: HTTP %d" filename status))))))

(defn -main [& args]
  (let [repo (or (first args) "google/gemma-4-E2B-it")
        model-name (last (str/split repo #"/"))
        target-dir (str ".models/" model-name)
        token (load-hf-token)
        client (http-client)
        _ (println (format "Fetching file manifest for [%s]..." repo))
        files (fetch-repo-files client repo token)
        relevant-files (filter-relevant-files files)]
    (if (empty? relevant-files)
      (println (format "No files found for [%s]. If restricted, ensure HF_TOKEN is set." repo))
      (do
        (println (format "Found %d model files. Downloading to [%s]..." (count relevant-files) target-dir))
        (doseq [f relevant-files]
          (download-file client repo f target-dir token))
        (println "\n=== Download Complete! ===")))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
