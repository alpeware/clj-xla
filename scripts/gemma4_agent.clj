(ns scripts.gemma4-agent
  "Autonomous software architecture agent loop powered by Gemma 4, XLA execution, and SCI Clojure tool calling."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clj-xla.core :as xla]
            [scripts.gemma4-inference :as gemma4-inf]
            [sci.core :as sci]))

(def DEFAULT_SYSTEM_PROMPT
  "You are an autonomous software engineering agent equipped with a live Clojure execution environment (`sci`).
You have access to the following built-in Clojure functions:
- (list-files \".\") to list files in a directory
- (slurp \"path/to/file.clj\") to read file contents
- (system-info) to inspect system metadata
- Standard Clojure math and collection utilities (reduce, map, filter, range, +, *, etc.)

To invoke a tool, write a Clojure code block formatted as:
```clojure
(list-files \".\")
```
When you receive tool output, continue your task until complete.")

(def DEFAULT_AGENT_OPTS
  {:prompt "Inspect the files in the workspace and calculate the total size of Clojure source files."
   :system DEFAULT_SYSTEM_PROMPT
   :max-new-tokens 300
   :max-turns 5
   :temperature 0.0
   :top-k 10
   :backend :cpu
   :precision :bf16
   :out "scratch/output_agent_loop.txt"
   :profile-out "scratch/gemma4_agent_profile.edn"
   :chrome-trace-out "scratch/gemma4_agent_chrome_trace.json"
   :quiet false})

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` or ```clj ... ``` code block strings from text, cleaning nested backtick artifacts."
  [text]
  (let [pattern #"(?s)```(?:clojure|clj)\s*\n(.*?)(?:```|$)"
        raw-matches (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern text))))]
    (mapv (fn [block]
            (-> block
                (str/replace #"^```[a-z]*>?" "")
                (str/replace #"```$" "")
                str/trim))
          raw-matches)))

(defn create-agent-sci-ctx
  "Creates a safe SCI sandbox context populated with useful Clojure agent helper functions."
  []
  (sci/init
   {:bindings {'println println
               'print print
               'prn prn
               'str str
               'slurp (fn [f] (try (slurp f) (catch Exception e (str "Error reading file: " (.getMessage e)))))
               'spit (fn [f c] (try (spit f c) (str "Successfully wrote to " f) (catch Exception e (str "Error writing file: " (.getMessage e)))))
               'list-files (fn [dir]
                             (try
                               (let [d (io/file dir)]
                                 (mapv (fn [^java.io.File f]
                                         {:name (.getName f)
                                          :dir? (.isDirectory f)
                                          :size (.length f)})
                                       (.listFiles d)))
                               (catch Exception e (str "Error listing files: " (.getMessage e)))))
               'system-info (fn []
                              {:os (System/getProperty "os.name")
                               :arch (System/getProperty "os.arch")
                               :java (System/getProperty "java.version")
                               :cpus (.availableProcessors (Runtime/getRuntime))
                               :free-mem (.freeMemory (Runtime/getRuntime))})}}))

(defn eval-tool-code
  "Evaluates `code-str` in the SCI sandbox and returns formatted execution result."
  [sci-ctx code-str]
  (try
    (let [out-writer (java.io.StringWriter.)
          eval-res (binding [*out* out-writer]
                     (sci/eval-string* sci-ctx code-str))
          printed (str out-writer)
          formatted-res (if (seq printed)
                          (str printed "\n=> " (pr-str eval-res))
                          (pr-str eval-res))]
      {:status :success :output formatted-res})
    (catch Exception e
      {:status :error :output (str "Execution Exception: " (.getMessage e))})))

(defn parse-agent-cli-args
  "Parses CLI flags for gemma4_agent."
  [args]
  (loop [remaining args
         opts DEFAULT_AGENT_OPTS]
    (if (empty? remaining)
      opts
      (let [arg (first remaining)
            more (rest remaining)]
        (cond
          (= arg "--prompt") (recur (rest more) (assoc opts :prompt (first more)))
          (= arg "--prompt-file") (recur (rest more) (assoc opts :prompt (slurp (first more))))
          (= arg "--system") (recur (rest more) (assoc opts :system (first more)))
          (= arg "--system-file") (recur (rest more) (assoc opts :system (slurp (first more))))
          (= arg "--max-turns") (recur (rest more) (assoc opts :max-turns (Integer/parseInt (first more))))
          (= arg "--max-new-tokens") (recur (rest more) (assoc opts :max-new-tokens (Integer/parseInt (first more))))
          (= arg "--backend") (recur (rest more) (assoc opts :backend (keyword (first more))))
          (= arg "--out") (recur (rest more) (assoc opts :out (first more)))
          (= arg "--model") (recur (rest more) (assoc opts :model (first more)))
          :else (recur more opts))))))

(defn format-agent-chat-prompt
  "Formats conversation history into Gemma 4 Turn syntax."
  [system-prompt history]
  (let [sys-turn (if (seq system-prompt)
                   (str "<|turn>system\n" system-prompt "\n<turn|>\n")
                   "")]
    (str "<bos>" sys-turn
         (str/join "" (map (fn [{:keys [role content]}]
                             (str "<|turn>" (name role) "\n" content "\n<turn|>\n"))
                           history))
         "<|turn>model\n")))

(defn run-agent-loop
  "Runs autonomous agent loop with SCI Clojure tool calling across multiple turns."
  [session initial-prompt]
  (let [{:keys [opts]} session
        {:keys [system max-turns out quiet]} opts
        sci-ctx (create-agent-sci-ctx)
        history (atom [{:role :user :content initial-prompt}])
        transcript (atom [])]
    (loop [turn 1]
      (if (> turn max-turns)
        (do
          (when-not quiet (println (format "\n[Agent] Reached max-turns limit (%d)." max-turns)))
          @transcript)
        (do
          (when-not quiet (println (format "\n==================================================")))
          (when-not quiet (println (format "=== Agent Turn %d/%d ===" turn max-turns)))
          (when-not quiet (println (format "==================================================")))
          (let [formatted-prompt (format-agent-chat-prompt system @history)
                _ (when-not quiet (println "Executing Gemma 4 Agent Forward Pass..."))
                full-gen (gemma4-inf/generate-text-string session formatted-prompt)
                model-text (if (re-find #"<\|turn>model" full-gen)
                             (last (str/split full-gen #"<\|turn>model\r?\n?"))
                             full-gen)
                model-reply (str/trim (str/replace model-text #"<bos>|<eos>|<turn\|>|<\|turn>" ""))
                code-blocks (extract-clojure-code-blocks model-reply)]

            (swap! history conj {:role :model :content model-reply})
            (swap! transcript conj {:turn turn :role :model :content model-reply})

            (if (empty? code-blocks)
              (do
                (when-not quiet (println "\n[Agent] No further tool calls requested. Task completed!"))
                (when (seq out)
                  (spit out (str/join "\n\n" (map :content @transcript)))
                  (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
                @transcript)

              (let [tool-code (first code-blocks)
                    _ (when-not quiet
                        (println "\n--------------------------------------------------")
                        (println "[Agent Tool Call (SCI Clojure)]:")
                        (println tool-code)
                        (println "--------------------------------------------------"))
                    eval-res (eval-tool-code sci-ctx tool-code)
                    obs-str (str "Tool Execution Observation:\n" (:output eval-res))]

                (when-not quiet
                  (println "\n[Tool Observation Output]:")
                  (println (:output eval-res)))

                (swap! history conj {:role :user :content obs-str})
                (swap! transcript conj {:turn turn :role :tool :content obs-str})
                (recur (inc turn))))))))))

(defn -main
  "CLI Entrypoint for Gemma 4 Agent."
  [& args]
  (try
    (let [opts (parse-agent-cli-args args)]
      (when (gemma4-inf/needs-libjsig-reexec? opts)
        (gemma4-inf/reexec-with-libjsig! args "scripts.gemma4-agent"))
      (let [model-dir (or (:model opts)
                          (first (filter #(.exists (io/file %)) gemma4-inf/DEFAULT_MODEL_DIRS)))
            _ (when-not model-dir
                (println "Error: Gemma 4 model directory not found.")
                (System/exit 1))
            session (gemma4-inf/init-inference-session opts)
            _ (when-not (:quiet opts) (println "Transferring Gemma 4 model weights to PJRT Device Memory..."))
            device-weights (gemma4-inf/allocate-device-weights session)
            session (assoc session :device-weights device-weights)]
        (try
          (run-agent-loop session (:prompt opts))
          (finally
            (doseq [w device-weights]
              (xla/destroy-buffer! (:ctx session) w))))))
    (catch Throwable e
      (println "\nAgent Exception:" (.getMessage e))
      (.printStackTrace e))
    (finally
      (.. Runtime getRuntime (halt 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
