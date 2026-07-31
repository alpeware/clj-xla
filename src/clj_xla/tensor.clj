(ns clj-xla.tensor
  "Shadowed operators and scalar auto-lifting for symbolic tensor operations."
  (:refer-clojure :exclude [+ * - /]))

(defrecord Tracer [id type])

(defn tracer?
  "Returns true if `x` is a Tracer instance."
  [x]
  (instance? Tracer x))

(def ^:dynamic *trace-ctx*
  "Thread-local dynamic variable bound to an atom holding the active trace state during tracing."
  nil)

(defn- gen-var-id! [prefix]
  (if *trace-ctx*
    (keyword (str (name prefix) "_" (swap! (:var-counter *trace-ctx*) inc)))
    (keyword (str (name prefix) "_" (gensym)))))

(defn emit-constant!
  "Emits an explicit SSA constant equation if `x` is a number and returns a new Tracer."
  [x target-type]
  (if (tracer? x)
    x
    (let [cid (gen-var-id! "c")
          eqn {:op :stablehlo/constant :value x :outvars [cid]}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer cid (or target-type [:tensor [] :f32])))))

(defn- emit-binary-op! [op a b]
  (if (or (tracer? a) (tracer? b) (some? *trace-ctx*))
    (let [target-type (or (when (tracer? a) (:type a))
                          (when (tracer? b) (:type b))
                          [:tensor [] :f32])
          ta (emit-constant! a target-type)
          tb (emit-constant! b target-type)
          out-id (gen-var-id! "t")
          eqn {:op op :invars [(:id ta) (:id tb)] :outvars [out-id]}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer out-id target-type))
    (cond
      (= op :stablehlo/add) (clojure.core/+ a b)
      (= op :stablehlo/subtract) (clojure.core/- a b)
      (= op :stablehlo/multiply) (clojure.core/* a b)
      (= op :stablehlo/divide) (clojure.core// a b)
      (= op :stablehlo/power) (Math/pow (double a) (double b))
      :else (throw (ex-info "Unknown binary op" {:op op})))))

(defn- emit-unary-op! [op a]
  (if (or (tracer? a) (some? *trace-ctx*))
    (let [ta (emit-constant! a nil)
          target-type (:type ta)
          out-id (gen-var-id! "t")
          eqn {:op op :invars [(:id ta)] :outvars [out-id]}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer out-id target-type))
    (cond
      (= op :stablehlo/tanh) (Math/tanh (double a))
      (= op :stablehlo/sqrt) (Math/sqrt (double a))
      :else (throw (ex-info "Unknown unary op" {:op op})))))

;; Shadowed Clojure Operators

(defn +
  "Shadowed addition operator supporting Tensors and scalars."
  ([a] a)
  ([a b] (emit-binary-op! :stablehlo/add a b))
  ([a b & more] (reduce + (+ a b) more)))

(defn -
  "Shadowed subtraction operator supporting Tensors and scalars."
  ([a] (emit-binary-op! :stablehlo/subtract 0.0 a))
  ([a b] (emit-binary-op! :stablehlo/subtract a b))
  ([a b & more] (reduce - (- a b) more)))

(defn *
  "Shadowed multiplication operator supporting Tensors and scalars."
  ([a] a)
  ([a b] (emit-binary-op! :stablehlo/multiply a b))
  ([a b & more] (reduce * (* a b) more)))

(defn /
  "Shadowed division operator supporting Tensors and scalars."
  ([a] (emit-binary-op! :stablehlo/divide 1.0 a))
  ([a b] (emit-binary-op! :stablehlo/divide a b))
  ([a b & more] (reduce / (/ a b) more)))

(defn pow
  "Shadowed elementwise exponentiation (power)."
  [a b]
  (emit-binary-op! :stablehlo/power a b))

(defn tanh
  "Shadowed elementwise hyperbolic tangent."
  [a]
  (emit-unary-op! :stablehlo/tanh a))

(defn sqrt
  "Shadowed elementwise square root."
  [a]
  (emit-unary-op! :stablehlo/sqrt a))

(defn reduce-mean
  "Computes tensor mean reduction across specified axes."
  [x & {:keys [axes keep-dims] :or {axes [-1] keep-dims true}}]
  (if (or (tracer? x) (some? *trace-ctx*))
    (let [tx (emit-constant! x nil)
          target-type (:type tx)
          out-id (gen-var-id! "t")
          eqn {:op :stablehlo/reduce_mean
               :invars [(:id tx)]
               :outvars [out-id]
               :attrs {:axes axes :keep_dims keep-dims}}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer out-id target-type))
    x))
