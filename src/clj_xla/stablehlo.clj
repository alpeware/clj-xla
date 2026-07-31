(ns clj-xla.stablehlo
  "EDN SSA graph Malli schemas and StableHLO MLIR text formatter."
  (:require [clojure.string :as str]
            [malli.core :as m]))

;; -----------------------------------------------------------------------------
;; Malli Schemas for Jaxpr-Style EDN SSA Intermediate Representation
;; -----------------------------------------------------------------------------

(def ElementTypeSchema
  "Schema for supported tensor element data types."
  [:enum :f16 :f32 :f64 :bf16 :i8 :i16 :i32 :i64 :pred])

(def TensorTypeSchema
  "Schema for a tensor type specifier tuple [:tensor dims dtype]."
  [:tuple [:= :tensor] [:vector :int] ElementTypeSchema])

(def VarBindingSchema
  "Schema for a named input variable binding [var-name tensor-type]."
  [:tuple keyword? TensorTypeSchema])

(def EquationSchema
  "Schema for a single SSA graph equation."
  [:map
   [:op keyword?]
   [:invars {:optional true} [:vector keyword?]]
   [:outvars [:vector keyword?]]
   [:value {:optional true} [:or number? boolean? vector?]]
   [:attrs {:optional true} map?]])

(def GraphSchema
  "Schema for an EDN SSA computation graph map."
  [:map
   [:name string?]
   [:invars [:vector VarBindingSchema]]
   [:outvars [:vector keyword?]]
   [:eqns [:vector EquationSchema]]])

(defn validate-graph
  "Validates `graph` against `GraphSchema`. Throws an exception if invalid."
  [graph]
  (if (m/validate GraphSchema graph)
    graph
    (throw (ex-info "Invalid EDN SSA Graph schema"
                    {:explanation (m/explain GraphSchema graph)}))))

;; -----------------------------------------------------------------------------
;; MLIR Text Serialization
;; -----------------------------------------------------------------------------

(defn type->mlir-string
  "Converts `[:tensor [1 128 768] :f32]` to `tensor<1x128x768xf32>`."
  [tensor-type]
  (if (vector? tensor-type)
    (let [[_ dims dtype] tensor-type
          dim-str (str/join "x" dims)
          dtype-str (name dtype)]
      (if (seq dims)
        (str "tensor<" dim-str "x" dtype-str ">")
        (str "tensor<" dtype-str ">")))
    "tensor<f32>"))

(defn- format-op-name [op]
  (let [op-str (name op)]
    (str "stablehlo." op-str)))

(defn- infer-var-types [invars eqns]
  (let [initial-types (into {} (map (fn [[v t]] [v (type->mlir-string t)]) invars))]
    (reduce (fn [acc {:keys [invars outvars]}]
              (let [in-vars (or invars [])
                    in-types (keep acc in-vars)
                    out-types (keep acc outvars)
                    tensor-type (or (first (filter #(not= % "tensor<f32>") (concat in-types out-types)))
                                    (first in-types)
                                    (first out-types)
                                    "tensor<f32>")
                    acc' (reduce (fn [a v] (assoc a v tensor-type)) acc outvars)]
                (reduce (fn [a v]
                          (if (or (not (contains? a v)) (= (get a v) "tensor<f32>"))
                            (assoc a v tensor-type)
                            a))
                        acc'
                        in-vars)))
            initial-types
            eqns)))

(defn- format-equation [eqn var-types]
  (let [{:keys [op invars outvars value _attrs]} (assoc eqn :invars (or (:invars eqn) []))
        mlir-op (format-op-name op)
        out-var (first outvars)]
    (cond
      (= op :stablehlo/constant)
      (let [val-str (if (float? value)
                      (format "%.6e" value)
                      (str value))
            out-type (get var-types out-var "tensor<f32>")]
        (str "    %" (name out-var) " = " mlir-op " dense<" val-str "> : " out-type))

      :else
      (let [in-vars-str (str/join ", " (map #(str "%" (name %)) invars))
            in-types (map #(get var-types % "tensor<f32>") invars)
            out-type (get var-types out-var "tensor<f32>")
            in-types-str (str/join ", " in-types)]
        (str "    %" (name out-var) " = " mlir-op " " in-vars-str " : (" in-types-str ") -> " out-type)))))

(defn graph->mlir-text
  "Converts a validated EDN SSA graph into a formatted StableHLO MLIR textual module."
  [graph]
  (validate-graph graph)
  (let [{:keys [name invars outvars eqns]} graph
        var-types (infer-var-types invars eqns)
        param-strs (map (fn [[v t]] (str "%" (clojure.core/name v) ": " (type->mlir-string t))) invars)
        fn-args-str (str/join ", " param-strs)
        return-types (map #(get var-types % "tensor<f32>") outvars)
        fn-ret-str (str "(" (str/join ", " return-types) ")")
        eqn-lines (map #(format-equation % var-types) eqns)
        return-vars-str (str/join ", " (map #(str "%" (clojure.core/name %)) outvars))
        return-types-str (str/join ", " return-types)]
    (str "module @" name " {\n"
         "  func.func @main(" fn-args-str ") -> " fn-ret-str " {\n"
         (str/join "\n" eqn-lines) "\n"
         "    return " return-vars-str " : " return-types-str "\n"
         "  }\n"
         "}\n")))
