(ns scripts.download-hf
  "Download Hugging Face model weights using pure Clojure and JDK HttpClient."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.net URI]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest HttpResponse$BodyHandlers]
           [java.nio.file CopyOption Files Paths StandardCopyOption]))

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
  "Filters repository files to target model weights, tokenizers, and configs."
  [files]
  (filterv #(re-find #"(safetensors|json|txt|model)$" %) files))

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
        req (cond-> (HttpRequest/newBuilder)
              true (.uri (URI/create url))
              token (.header "Authorization" (str "Bearer " token))
              true (.GET)
              true (.build))]
    (println (format "  --> Downloading %s..." filename))
    (let [resp (.send client req (HttpResponse$BodyHandlers/ofInputStream))
          in-stream (.body resp)
          options (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING])]
      (Files/copy ^java.io.InputStream in-stream ^java.nio.file.Path out-path ^"[Ljava.nio.file.CopyOption;" options)
      (println (format "      Done: %s (%d MB)"
                       (.getFileName out-path)
                       (quot (Files/size out-path) (* 1024 1024)))))))

(defn -main [& args]
  (let [repo (or (first args) "google/gemma-2-2b-it")
        model-name (last (str/split repo #"/"))
        target-dir (str ".models/" model-name)
        token (load-hf-token)]
    (if-not token
      (println "Error: No HF token found in .env map or HF_TOKEN env var!")
      (let [client (http-client)
            _ (println (format "Fetching file manifest for [%s]..." repo))
            files (fetch-repo-files client repo token)
            relevant-files (filter-relevant-files files)]
        (println (format "Found %d model files. Downloading to [%s]..." (count relevant-files) target-dir))
        (doseq [f relevant-files]
          (download-file client repo f target-dir token))
        (println "\n=== Download Complete! ===")))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
