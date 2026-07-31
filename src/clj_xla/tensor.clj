(ns clj-xla.tensor
  "Shadowed operators and scalar auto-lifting for symbolic tensor operations."
  (:refer-clojure :exclude [+ * - /]))

(defrecord Tracer [id type])

(defn tracer?
  "Returns true if `x` is a Tracer instance."
  [x]
  (or (instance? Tracer x)
      (and (map? x) (contains? x :id) (contains? x :type))))

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
  (let [target-type (or (when (tracer? a) (:type a))
                        (when (tracer? b) (:type b))
                        [:tensor [] :f32])
        ta (emit-constant! a target-type)
        tb (emit-constant! b target-type)
        out-id (gen-var-id! "t")
        eqn {:op op :invars [(:id ta) (:id tb)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id target-type)))

(defn- emit-unary-op! [op a]
  (let [ta (emit-constant! a nil)
        target-type (:type ta)
        out-id (gen-var-id! "t")
        eqn {:op op :invars [(:id ta)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id target-type)))

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

(defn exp
  "Shadowed elementwise exponential function."
  [a]
  (emit-unary-op! :stablehlo/exp a))

(defn tanh
  "Shadowed elementwise hyperbolic tangent."
  [a]
  (emit-unary-op! :stablehlo/tanh a))

(defn sqrt
  "Shadowed elementwise square root."
  [a]
  (emit-unary-op! :stablehlo/sqrt a))

(defn gather
  "Performs index-based table lookup.
   operand: [vocab_size, hidden_dim]
   start-indices: [batch, seq_len]
   Returns: [batch, seq_len, hidden_dim]"
  [operand start-indices]
  (let [t-op (emit-constant! operand nil)
        t-idx (emit-constant! start-indices nil)
        [_kw op-shape dtype] (:type t-op)
        [_kw2 idx-shape _] (:type t-idx)
        hidden-dim (last op-shape)
        out-shape (conj (vec idx-shape) hidden-dim)
        out-id (gen-var-id! "t_gather")
        eqn {:op :stablehlo/gather
             :invars [(:id t-op) (:id t-idx)]
             :outvars [out-id]
             :attrs {:offset_dims [(clojure.core/dec (clojure.core/count out-shape))]
                     :collapsed_slice_dims [0]
                     :start_index_map [0]
                     :index_vector_dim (clojure.core/count idx-shape)
                     :slice_sizes [1 hidden-dim]}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id [:tensor out-shape dtype])))

(defn dot-general
  "General matrix multiplication and tensor contraction.
   attrs map requires :contracting_dims and optional :batch_dims."
  [lhs rhs attrs]
  (let [tlhs (emit-constant! lhs nil)
        trhs (emit-constant! rhs nil)
        [_kw lhs-shape dtype] (:type tlhs)
        [_kw2 rhs-shape _] (:type trhs)
        out-id (gen-var-id! "t_dot")
        c-lhs (or (get-in attrs [:contracting_dims :lhs])
                  (get-in attrs [:contracting-dims :lhs])
                  [(clojure.core/dec (clojure.core/count lhs-shape))])
        c-rhs (or (get-in attrs [:contracting_dims :rhs])
                  (get-in attrs [:contracting-dims :rhs])
                  [0])
        b-lhs (or (get-in attrs [:batch_dims :lhs])
                  (get-in attrs [:batch-dims :lhs])
                  [])
        b-rhs (or (get-in attrs [:batch_dims :rhs])
                  (get-in attrs [:batch-dims :rhs])
                  [])
        c-lhs-set (set c-lhs)
        c-rhs-set (set c-rhs)
        b-lhs-set (set b-lhs)
        b-rhs-set (set b-rhs)
        batch-dims (mapv #(nth lhs-shape %) b-lhs)
        lhs-free (keep-indexed (fn [idx dim] (when-not (or (contains? c-lhs-set idx) (contains? b-lhs-set idx)) dim)) lhs-shape)
        rhs-free (keep-indexed (fn [idx dim] (when-not (or (contains? c-rhs-set idx) (contains? b-rhs-set idx)) dim)) rhs-shape)
        out-shape (vec (concat batch-dims lhs-free rhs-free))
        eqn {:op :stablehlo/dot_general
             :invars [(:id tlhs) (:id trhs)]
             :outvars [out-id]
             :attrs attrs}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id [:tensor out-shape dtype])))

(defn matmul
  "Standard 2D/3D matrix multiplication (wrapper around dot-general)."
  [a b]
  (let [ta (emit-constant! a nil)
        tb (emit-constant! b nil)
        [_kw a-shape _] (:type ta)
        [_kw2 b-shape _] (:type tb)
        a-rank (clojure.core/count a-shape)
        b-rank (clojure.core/count b-shape)
        lhs-c [(clojure.core/dec a-rank)]
        rhs-c (if (clojure.core/>= b-rank 2) [(clojure.core/- b-rank 2)] [0])]
    (dot-general ta tb {:contracting_dims {:lhs lhs-c :rhs rhs-c}})))

(defn reshape
  "Reshapes input tensor `x` to `new-shape`."
  [x new-shape]
  (let [tx (emit-constant! x nil)
        [_kw _shape dtype] (:type tx)
        out-id (gen-var-id! "t_reshape")
        eqn {:op :stablehlo/reshape
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:shape (vec new-shape)}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id [:tensor (vec new-shape) dtype])))

(defn transpose
  "Permutes dimensions of tensor `x` according to `permutation` vector."
  [x permutation]
  (let [tx (emit-constant! x nil)
        [_kw shape dtype] (:type tx)
        new-shape (mapv #(nth shape %) permutation)
        out-id (gen-var-id! "t_transpose")
        eqn {:op :stablehlo/transpose
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:permutation (vec permutation)}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id [:tensor new-shape dtype])))

(defn- extract-num [v]
  (cond
    (number? v) (long v)
    :else 0))

(defn slice
  "Extracts a sub-tensor slice from `x` between `start-indices` and `limit-indices` using `strides`."
  [x start-indices limit-indices strides]
  (let [tx (emit-constant! x nil)
        [_kw shape dtype] (:type tx)
        rank (clojure.core/count shape)
        starts (mapv extract-num start-indices)
        limits (mapv extract-num limit-indices)
        step-strides (or (when (seq strides) (mapv extract-num strides)) (vec (repeat rank 1)))
        out-shape (mapv (fn [i]
                          (let [s (nth starts i)
                                l (nth limits i)
                                st (nth step-strides i)]
                            (long (Math/ceil (clojure.core// (double (clojure.core/- l s)) (double st))))))
                        (range rank))
        out-id (gen-var-id! "t_slice")
        eqn {:op :stablehlo/slice
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:start_indices starts
                     :limit_indices limits
                     :strides step-strides}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id [:tensor out-shape dtype])))

(defn reduce-sum
  "Computes tensor sum reduction across specified axes."
  [x & {:keys [axes keep-dims] :or {axes [-1] keep-dims true}}]
  (let [tx (emit-constant! x nil)
        [t-kw shape dtype] (:type tx)
        rank (clojure.core/count shape)
        norm-axes (mapv #(if (clojure.core/neg? %) (clojure.core/+ rank %) %) (or axes [-1]))
        axes-set (set norm-axes)
        out-shape (mapv (fn [idx dim]
                          (if (contains? axes-set idx)
                            (if keep-dims 1 nil)
                            dim))
                        (range rank)
                        shape)
        out-shape-clean (vec (remove nil? out-shape))
        out-type [t-kw out-shape-clean dtype]
        out-id (gen-var-id! "t_sum")
        eqn {:op :stablehlo/reduce_sum
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:axes norm-axes :keep_dims keep-dims}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn reduce-max
  "Computes tensor max reduction across specified axes."
  [x & {:keys [axes keep-dims] :or {axes [-1] keep-dims true}}]
  (let [tx (emit-constant! x nil)
        [t-kw shape dtype] (:type tx)
        rank (clojure.core/count shape)
        norm-axes (mapv #(if (clojure.core/neg? %) (clojure.core/+ rank %) %) (or axes [-1]))
        axes-set (set norm-axes)
        out-shape (mapv (fn [idx dim]
                          (if (contains? axes-set idx)
                            (if keep-dims 1 nil)
                            dim))
                        (range rank)
                        shape)
        out-shape-clean (vec (remove nil? out-shape))
        out-type [t-kw out-shape-clean dtype]
        out-id (gen-var-id! "t_max")
        eqn {:op :stablehlo/reduce_max
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:axes norm-axes :keep_dims keep-dims}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn reduce-mean
  "Computes tensor mean reduction across specified axes by dividing reduce-sum by dimension size."
  [x & {:keys [axes keep-dims] :or {axes [-1] keep-dims true}}]
  (let [tx (emit-constant! x nil)
        [_kw shape _dtype] (:type tx)
        rank (clojure.core/count shape)
        norm-axes (mapv #(if (clojure.core/neg? %) (clojure.core/+ rank %) %) (or axes [-1]))
        red-count (double (reduce clojure.core/* (map #(nth shape %) norm-axes)))
        sum-tracer (reduce-sum tx :axes norm-axes :keep-dims keep-dims)]
    (/ sum-tracer red-count)))
