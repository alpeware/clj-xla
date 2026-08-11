(ns clj-xla.tensor
  "Symbolic execution tracer and high-level Clojure Tensor Operator primitives."
  (:refer-clojure :exclude [+ * - /]))

(defrecord Tracer [id type])

(def ^:dynamic *trace-ctx* nil)

(defn tracer?
  "Checks if val is a Tracer record."
  [val]
  (instance? Tracer val))

(defn- gen-var-id! [prefix]
  (if *trace-ctx*
    (keyword (str prefix "_" (swap! (:var-counter *trace-ctx*) inc)))
    (keyword (str prefix "_" (gensym)))))

(defn emit-constant!
  "Emits a :stablehlo/constant equation into *trace-ctx* if val is a primitive number."
  [val target-type]
  (cond
    (tracer? val)
    val

    (number? val)
    (let [out-id (gen-var-id! "c")
          out-type (or target-type [:tensor [] :f32])
          eqn {:op :stablehlo/constant :outvars [out-id] :value val :type out-type}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer out-id out-type))

    (vector? val)
    (let [out-id (gen-var-id! "c_vec")
          out-type (or target-type [:tensor [128 128] :f32])
          eqn {:op :stablehlo/constant :outvars [out-id] :value val :type out-type}]
      (when *trace-ctx*
        (swap! (:eqns *trace-ctx*) conj eqn))
      (->Tracer out-id out-type))

    :else
    (throw (ex-info "Cannot emit constant for unsupported value" {:value val}))))

(declare broadcast-in-dim)

(defn- broadcast-shapes [shape-a shape-b]
  (let [rank-a (clojure.core/count shape-a)
        rank-b (clojure.core/count shape-b)
        max-rank (clojure.core/max rank-a rank-b)
        pad-a (vec (clojure.core/concat (repeat (clojure.core/- max-rank rank-a) 1) shape-a))
        pad-b (vec (clojure.core/concat (repeat (clojure.core/- max-rank rank-b) 1) shape-b))]
    (mapv clojure.core/max pad-a pad-b)))

(defn- binary-elementwise-op [op-kw prefix a b]
  (let [ta (emit-constant! a nil)
        tb (emit-constant! b (:type ta))
        [t-kw shape-a dtype-a] (:type ta)
        [_ shape-b _] (:type tb)
        out-shape (broadcast-shapes shape-a shape-b)
        rank-out (count out-shape)
        ta-bcast (if (= shape-a out-shape)
                   ta
                   (let [rank-a (count shape-a)
                         bcast-dims (vec (map int (range (clojure.core/- rank-out rank-a) rank-out)))]
                     (broadcast-in-dim ta out-shape bcast-dims)))
        tb-bcast (if (= shape-b out-shape)
                   tb
                   (let [rank-b (count shape-b)
                         bcast-dims (vec (map int (range (clojure.core/- rank-out rank-b) rank-out)))]
                     (broadcast-in-dim tb out-shape bcast-dims)))
        out-type [t-kw out-shape dtype-a]
        out-id (gen-var-id! prefix)
        eqn {:op op-kw :invars [(:id ta-bcast) (:id tb-bcast)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn +
  "Elementwise tensor addition."
  ([a b]
   (binary-elementwise-op :stablehlo/add "t_add" a b))
  ([a b & more]
   (reduce + (+ a b) more)))

(defn -
  "Elementwise tensor subtraction or unary negation."
  ([a]
   (binary-elementwise-op :stablehlo/subtract "t_sub" 0.0 a))
  ([a b]
   (binary-elementwise-op :stablehlo/subtract "t_sub" a b))
  ([a b & more]
   (reduce - (- a b) more)))

(defn *
  "Elementwise tensor multiplication."
  ([a b]
   (binary-elementwise-op :stablehlo/multiply "t_mul" a b))
  ([a b & more]
   (reduce * (* a b) more)))

(defn /
  "Elementwise tensor division."
  ([a b]
   (binary-elementwise-op :stablehlo/divide "t_div" a b))
  ([a b & more]
   (reduce / (/ a b) more)))

(defn exp
  "Elementwise natural exponential function."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_exp")
        eqn {:op :stablehlo/exp :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn sqrt
  "Elementwise square root function."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_sqrt")
        eqn {:op :stablehlo/sqrt :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn rsqrt
  "Elementwise reciprocal square root: 1 / sqrt(x)."
  [x]
  (/ 1.0 (sqrt x)))

(defn pow
  "Elementwise power function."
  [x p]
  (cond
    (= p 2.0) (* x x)
    (= p 3.0) (* x (* x x))
    :else (let [tx (emit-constant! x nil)
                out-id (gen-var-id! "t_pow")
                eqn {:op :stablehlo/power :invars [(:id tx)] :outvars [out-id] :value p}]
            (when *trace-ctx*
              (swap! (:eqns *trace-ctx*) conj eqn))
            (->Tracer out-id (:type tx)))))

(defn maximum
  "Elementwise maximum function."
  [x y]
  (binary-elementwise-op :stablehlo/maximum "t_max" x y))

(defn tanh
  "Elementwise hyperbolic tangent function."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_tanh")
        eqn {:op :stablehlo/tanh :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn sigmoid
  "Elementwise logistic sigmoid function: 1 / (1 + exp(-x)). Emits native stablehlo/logistic."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_logistic")
        eqn {:op :stablehlo/logistic :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn sin
  "Elementwise sine function."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_sin")
        eqn {:op :stablehlo/sine :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn convert
  "Elementwise type conversion function."
  [x target-dtype]
  (let [tx (emit-constant! x nil)
        [t-kw shape _dtype] (:type tx)
        out-type [t-kw shape target-dtype]
        out-id (gen-var-id! "t_convert")
        eqn {:op :stablehlo/convert :invars [(:id tx)] :outvars [out-id] :attrs {:target_dtype target-dtype}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn cos
  "Elementwise cosine function."
  [x]
  (let [tx (emit-constant! x nil)
        out-id (gen-var-id! "t_cos")
        eqn {:op :stablehlo/cosine :invars [(:id tx)] :outvars [out-id]}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id (:type tx))))

(defn concatenate
  "Concatenates tensors along `dimension`."
  [tensors dimension]
  (let [tracers (mapv #(emit-constant! % nil) tensors)
        first-t (first tracers)
        [t-kw first-shape dtype] (:type first-t)
        rank (clojure.core/count first-shape)
        dim (if (neg? dimension) (clojure.core/+ rank dimension) dimension)
        total-dim-size (reduce clojure.core/+ (map (fn [tr] (nth (second (:type tr)) dim)) tracers))
        out-shape (assoc first-shape dim total-dim-size)
        out-type [t-kw out-shape dtype]
        out-id (gen-var-id! "t_concat")
        eqn {:op :stablehlo/concatenate
             :invars (mapv :id tracers)
             :outvars [out-id]
             :attrs {:dimension dim}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(def concat-tensor concatenate)

(defn dot-general
  "Batched matrix multiplication with explicit contracting and batching dimensions."
  [lhs rhs {:keys [contracting_dims batch_dims] :as attrs}]
  (let [tlhs (emit-constant! lhs nil)
        trhs (emit-constant! rhs nil)
        [t-kw lhs-shape dtype] (:type tlhs)
        [_ rhs-shape _] (:type trhs)
        c-lhs (set (get contracting_dims :lhs [(dec (clojure.core/count lhs-shape))]))
        c-rhs (set (get contracting_dims :rhs [0]))
        b-lhs (set (get batch_dims :lhs []))
        b-rhs (set (get batch_dims :rhs []))
        batch-dims (mapv #(nth lhs-shape %) (get batch_dims :lhs []))
        lhs-free (keep-indexed (fn [idx dim] (when-not (or (c-lhs idx) (b-lhs idx)) dim)) lhs-shape)
        rhs-free (keep-indexed (fn [idx dim] (when-not (or (c-rhs idx) (b-rhs idx)) dim)) rhs-shape)
        out-shape (vec (concat batch-dims lhs-free rhs-free))
        out-type [t-kw out-shape dtype]
        out-id (gen-var-id! "t_dot")
        eqn {:op :stablehlo/dot_general
             :invars [(:id tlhs) (:id trhs)]
             :outvars [out-id]
             :attrs attrs}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn matmul
  "Standard matrix multiplication (dot product over trailing dimension of lhs and leading dimension of rhs)."
  [lhs rhs]
  (let [tlhs (emit-constant! lhs nil)
        trhs (emit-constant! rhs nil)
        [_ lhs-shape _] (:type tlhs)
        rank (clojure.core/count lhs-shape)]
    (dot-general tlhs trhs {:contracting_dims {:lhs [(dec rank)] :rhs [0]}})))

(defn reshape
  "Reshapes tensor to new shape array."
  [x new-shape]
  (let [tx (emit-constant! x nil)
        [t-kw _shape dtype] (:type tx)
        out-type [t-kw new-shape dtype]
        out-id (gen-var-id! "t_reshape")
        eqn {:op :stablehlo/reshape
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:shape new-shape}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn transpose
  "Permutes dimensions of tensor according to permutation vector."
  [x permutation]
  (let [tx (emit-constant! x nil)
        [t-kw in-shape dtype] (:type tx)
        out-shape (mapv #(nth in-shape %) permutation)
        out-type [t-kw out-shape dtype]
        out-id (gen-var-id! "t_transpose")
        eqn {:op :stablehlo/transpose
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:permutation permutation}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn broadcast-in-dim
  "Broadcasts tracer x to new target shape and broadcast dimensions."
  [x target-shape bcast-dims]
  (let [tx (emit-constant! x nil)
        [t-kw _shape dtype] (:type tx)
        out-id (gen-var-id! "t_bcast")
        out-type [t-kw target-shape dtype]
        eqn {:op :stablehlo/broadcast_in_dim
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:broadcast_dimensions bcast-dims :target_shape target-shape}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn slice
  "Slices tensor along specified start, limit, and stride indices."
  [x start-indices limit-indices strides]
  (let [tx (emit-constant! x nil)
        [t-kw in-shape dtype] (:type tx)
        rank (clojure.core/count in-shape)
        out-shape (mapv (fn [i]
                          (let [s (nth start-indices i 0)
                                l (nth limit-indices i (nth in-shape i))
                                st (nth strides i 1)]
                            (long (Math/ceil (clojure.core// (double (clojure.core/- l s)) (double st))))))
                        (range rank))
        out-type [t-kw out-shape dtype]
        out-id (gen-var-id! "t_slice")
        eqn {:op :stablehlo/slice
             :invars [(:id tx)]
             :outvars [out-id]
             :attrs {:start_indices start-indices
                     :limit_indices limit-indices
                     :strides strides}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn dynamic-update-slice
  "Updates `operand` tensor with `update` slice starting at `start-indices`."
  [operand update start-indices]
  (let [t-op (emit-constant! operand nil)
        t-up (emit-constant! update nil)
        [t-kw in-shape dtype] (:type t-op)
        out-type [t-kw in-shape dtype]
        out-id (gen-var-id! "t_dus")
        eqn {:op :stablehlo/dynamic_update_slice
             :invars [(:id t-op) (:id t-up)]
             :outvars [out-id]
             :attrs {:start_indices start-indices}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

(defn gather
  "Gathers slices from operand at specified start-indices."
  [operand start-indices]
  (let [t-op (emit-constant! operand nil)
        raw-idx (emit-constant! start-indices nil)
        [op-kw op-shape dtype] (:type t-op)
        [_ idx-shape _] (:type raw-idx)
        idx-rank (clojure.core/count idx-shape)
        ;; Ensure start_indices is expanded so its last dimension is index_vector_dim (size 1)
        t-idx (cond
                (= idx-rank 2) (reshape raw-idx [(nth idx-shape 0) (nth idx-shape 1) 1])
                (= idx-rank 1) (reshape raw-idx [(nth idx-shape 0) 1])
                :else raw-idx)
        [_ final-idx-shape _] (:type t-idx)
        final-idx-rank (clojure.core/count final-idx-shape)
        hidden-dim (last op-shape)
        out-shape (conj idx-shape hidden-dim)
        out-type [op-kw out-shape dtype]
        out-id (gen-var-id! "t_gather")
        eqn {:op :stablehlo/gather
             :invars [(:id t-op) (:id t-idx)]
             :outvars [out-id]
             :attrs {:offset_dims [(clojure.core/dec final-idx-rank)]
                     :collapsed_slice_dims [0]
                     :start_index_map [0]
                     :index_vector_dim (clojure.core/dec final-idx-rank)
                     :slice_sizes [1 hidden-dim]}}]
    (when *trace-ctx*
      (swap! (:eqns *trace-ctx*) conj eqn))
    (->Tracer out-id out-type)))

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
