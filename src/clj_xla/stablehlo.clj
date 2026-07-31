(ns clj-xla.stablehlo
  "EDN SSA Graph Schema Definition, Validation, and MLIR Serialization."
  (:require [clojure.string :as str]
            [malli.core :as m]))

(def TensorType
  [:cat [:enum :tensor] [:vector :int] [:enum :f32 :f64 :i32 :i64 :f16 :bf16 :i8 :i16 :u8 :u16 :u32 :u64]])

(def VariableDecl
  [:tuple :keyword TensorType])

(def Equation
  [:map
   [:op keyword?]
   [:invars {:optional true} [:vector keyword?]]
   [:outvars [:vector keyword?]]
   [:value {:optional true} any?]
   [:attrs {:optional true} map?]])

(def GraphSchema
  [:map
   [:name string?]
   [:invars [:vector VariableDecl]]
   [:outvars [:vector keyword?]]
   [:eqns [:vector Equation]]])

(defn validate-graph
  "Validates graph structure against Malli GraphSchema."
  [graph]
  (if (m/validate GraphSchema graph)
    graph
    (throw (ex-info "Invalid SSA Graph EDN schema"
                    {:errors (m/explain GraphSchema graph)}))))

(defn type->mlir-string
  "Converts EDN SSA tensor type tuple to MLIR type string."
  [[_dims shape dtype]]
  (let [dtype-str (name dtype)
        dims-str (str/join "x" shape)]
    (if (seq shape)
      (str "tensor<" dims-str "x" dtype-str ">")
      (str "tensor<" dtype-str ">"))))

(defn- format-op-name [op-kw]
  (let [n (name op-kw)]
    (if (str/includes? n "/")
      n
      (str "stablehlo." n))))

(defn- find-tensor-type [invars var-types]
  (some (fn [inv]
          (when-let [t (get var-types inv)]
            (when (re-find #"^tensor<\d+x" t)
              t)))
        invars))

(defn- infer-var-types [invars eqns]
  (let [initial-types (into {} (map (fn [[v t]] [v (type->mlir-string t)]) invars))]
    (reduce (fn [acc {:keys [op invars outvars]}]
              (let [in-vars (or invars [])
                    non-scalar-t (find-tensor-type in-vars acc)
                    first-in (first in-vars)
                    in-type (or non-scalar-t (when first-in (get acc first-in)))]
                (cond
                  (= op :stablehlo/constant)
                  (assoc acc (first outvars) "tensor<f32>")

                  (= op :stablehlo/reduce_mean)
                  (assoc acc (first outvars) "tensor<1x128x1xf32>")

                  (= op :stablehlo/dot_general)
                  (let [lhs-v (first in-vars)
                        rhs-v (second in-vars)
                        lhs-t (get acc lhs-v "tensor<1x128x576xf32>")
                        rhs-t (get acc rhs-v "tensor<576x576xf32>")
                        lhs-k (or (second (re-find #"tensor<\d+x\d+x(\d+)xf32>" lhs-t)) "576")
                        [rhs-d0 rhs-d1] (or (rest (re-find #"tensor<(\d+)x(\d+)xf32>" rhs-t)) ["576" "576"])
                        out-dim (if (= rhs-d0 lhs-k) rhs-d1 rhs-d0)
                        out-t (str "tensor<1x128x" out-dim "xf32>")]
                    (assoc acc (first outvars) out-t))

                  :else
                  (let [target-type (or in-type "tensor<1x128x576xf32>")]
                    (reduce (fn [a v]
                              (if (contains? initial-types v)
                                a
                                (assoc a v target-type)))
                            acc
                            outvars)))))
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

      (= op :stablehlo/custom_call)
      (let [target-name (or (get attrs :call_target_name) "rope")
            in-var (first invars)
            in-type (get var-types in-var "tensor<1x128x576xf32>")
            out-type (get var-types out-var "tensor<1x128x576xf32>")]
        (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in-var) ") {call_target_name = \"" target-name "\"} : (" in-type ") -> " out-type))

      (= op :stablehlo/dot_general)
      (let [[lhs rhs] invars
            lhs-type (get var-types lhs "tensor<1x128x576xf32>")
            rhs-type (get var-types rhs "tensor<576x576xf32>")
            out-type (get var-types out-var "tensor<f32>")
            contracting (get attrs :contracting_dims [[2] [0]])
            rhs-c (str (first (second contracting)))]
        (str "    %" (name out-var) " = \"stablehlo.dot_general\"(%" (name lhs) ", %" (name rhs) ") {"
             "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [" rhs-c "]>} : "
             "(" lhs-type ", " rhs-type ") -> " out-type))

      :else
      (let [in-types (map #(get var-types % "tensor<1x128x576xf32>") invars)
            out-type (get var-types out-var (or (find-tensor-type invars var-types) (get var-types (first invars)) "tensor<1x128x576xf32>"))
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
