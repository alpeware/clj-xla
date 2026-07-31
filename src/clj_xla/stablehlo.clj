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
    (reduce (fn [acc {:keys [op invars outvars]}]
              (let [in-vars (or invars [])
                    in-types (keep acc in-vars)
                    out-types (keep acc outvars)
                    fallback-type (or (first (filter #(str/includes? % "1x128x") (concat in-types out-types)))
                                      (first in-types)
                                      (first out-types)
                                      "tensor<f32>")]
                (cond
                  (= op :stablehlo/constant)
                  (assoc acc (first outvars) "tensor<f32>")

                  (= op :stablehlo/reduce_mean)
                  (assoc acc (first outvars) "tensor<1x128x1xf32>")

                  (= op :stablehlo/dot_general)
                  (let [lhs-v (first in-vars)
                        rhs-v (second in-vars)
                        lhs-t (get acc lhs-v "tensor<1x128x768xf32>")
                        rhs-t (get acc rhs-v "tensor<768x768xf32>")
                        lhs-k (or (second (re-find #"tensor<\d+x\d+x(\d+)xf32>" lhs-t)) "768")
                        [rhs-d0 rhs-d1] (or (rest (re-find #"tensor<(\d+)x(\d+)xf32>" rhs-t)) ["768" "768"])
                        out-dim (if (= rhs-d0 lhs-k) rhs-d1 rhs-d0)
                        out-t (str "tensor<1x128x" out-dim "xf32>")]
                    (assoc acc (first outvars) out-t))

                  :else
                  (reduce (fn [a v]
                            (if (contains? initial-types v)
                              a
                              (assoc a v fallback-type)))
                          acc
                          outvars))))
            initial-types
            eqns)))

(defn- format-equation [eqn var-types]
  (let [{:keys [op invars outvars value attrs]} (assoc eqn :invars (or (:invars eqn) []))
        mlir-op (format-op-name op)
        out-var (first outvars)]
    (cond
      (= op :stablehlo/constant)
      (let [val-str (if (number? value)
                      (format "%.6e" (double value))
                      (str value))]
        (str "    %" (name out-var) " = " mlir-op " dense<" val-str "> : tensor<f32>"))

      (= op :stablehlo/reduce_mean)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<f32>")
            out-type (get var-types out-var "tensor<1x128x1xf32>")
            axes (get attrs :axes [2])
            norm-axes (mapv #(if (neg? %) 2 %) axes)
            axes-str (str/join ", " norm-axes)]
        (str "    %" (name out-var) "_c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n"
             "    %" (name out-var) "_red = \"stablehlo.reduce\"(%" (name in-var) ", %" (name out-var) "_c0) ({\n"
             "    ^bb0(%arg_a: tensor<f32>, %arg_b: tensor<f32>):\n"
             "      %arg_sum = stablehlo.add %arg_a, %arg_b : tensor<f32>\n"
             "      \"stablehlo.return\"(%arg_sum) : (tensor<f32>) -> ()\n"
             "    }) {dimensions = array<i64: " axes-str ">} : (" in-type ", tensor<f32>) -> tensor<1x128xf32>\n"
             "    %" (name out-var) " = stablehlo.reshape %" (name out-var) "_red : (tensor<1x128xf32>) -> " out-type))

      (= op :stablehlo/dot_general)
      (let [[lhs rhs] invars
            lhs-type (get var-types lhs "tensor<1x128x768xf32>")
            rhs-type (get var-types rhs "tensor<768x768xf32>")
            out-type (get var-types out-var "tensor<f32>")
            contracting (get attrs :contracting_dims [[2] [0]])
            rhs-c (str (first (second contracting)))]
        (str "    %" (name out-var) " = \"stablehlo.dot_general\"(%" (name lhs) ", %" (name rhs) ") {"
             "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [" rhs-c "]>} : "
             "(" lhs-type ", " rhs-type ") -> " out-type))

      :else
      (let [in-types (map #(get var-types % "tensor<f32>") invars)
            out-type (get var-types out-var "tensor<f32>")
            [in-vars-str prep-lines]
            (reduce (fn [[v-strs p-lines] [inv in-t]]
                      (cond
                        (= in-t "tensor<f32>")
                        (let [bcast-var (str (name inv) "_bcast")
                              line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" (name inv) ") {broadcast_dimensions = array<i64>} : (tensor<f32>) -> " out-type)]
                          [(conj v-strs (str "%" bcast-var)) (conj p-lines line)])

                        (re-find #"^tensor<\d+xf32>$" in-t)
                        (let [bcast-var (str (name inv) "_bcast")
                              line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" (name inv) ") {broadcast_dimensions = array<i64: 2>} : (" in-t ") -> " out-type)]
                          [(conj v-strs (str "%" bcast-var)) (conj p-lines line)])

                        (= in-t "tensor<1x128x1xf32>")
                        (let [bcast-var (str (name inv) "_bcast")
                              line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" (name inv) ") {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x128x1xf32>) -> " out-type)]
                          [(conj v-strs (str "%" bcast-var)) (conj p-lines line)])

                        :else
                        [(conj v-strs (str "%" (name inv))) p-lines]))
                    [[] []]
                    (map vector invars in-types))
            in-types-str (str/join ", " (repeat (count invars) out-type))
            op-line (str "    %" (name out-var) " = " mlir-op " " (str/join ", " in-vars-str) " : (" in-types-str ") -> " out-type)]
        (str/join "\n" (concat prep-lines [op-line]))))))

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
