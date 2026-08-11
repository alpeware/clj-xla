(ns clj-xla.benchmark.workloads
  "Standard benchmark workload specifications (GEMM, RMSNorm, SwiGLU, GQA Attention, GPT-2 block, Gemma 4 block)."
  (:require [clj-xla.benchmark.core :as bcore]
            [clj-xla.models.gemma :as gemma]
            [clj-xla.models.gpt2 :as gpt2]
            [clj-xla.nn.activations :refer [swiglu]]
            [clj-xla.nn.norm :refer [rms-norm]]
            [clj-xla.tensor :as t]
            [clj-xla.trace :refer [trace-graph]]))

(defn build-gemm-graph
  [m n k dtype]
  (trace-graph (str "gemm_" (name dtype) "_" m "_" n "_" k)
               [[:a [:tensor [m k] dtype]]
                [:b [:tensor [k n] dtype]]]
               (fn [a b] (t/matmul a b))))

(defn make-gemm-inputs
  [m n k dtype]
  (let [num-a (* m k)
        num-b (* k n)]
    (if (= dtype :bf16)
      {:a (short-array (repeat num-a (short 16256)))
       :b (short-array (repeat num-b (short 16256)))}
      {:a (float-array (repeat num-a 1.0))
       :b (float-array (repeat num-b 1.0))})))

(defn build-rms-norm-graph
  [batch seq-len dim dtype]
  (trace-graph (str "rms_norm_" batch "_" seq-len "_" dim)
               [[:x [:tensor [batch seq-len dim] dtype]]
                [:w [:tensor [dim] dtype]]]
               (fn [x w] (rms-norm x w 1e-6))))

(defn make-rms-norm-inputs
  [batch seq-len dim _dtype]
  {:x (float-array (repeat (* batch seq-len dim) 1.0))
   :w (float-array (repeat dim 1.0))})

(defn build-swiglu-graph
  [batch seq-len dim dtype]
  (let [inter (* 4 dim)]
    (trace-graph (str "swiglu_" batch "_" seq-len "_" dim)
                 [[:x [:tensor [batch seq-len dim] dtype]]
                  [:gate_w [:tensor [dim inter] dtype]]
                  [:up_w [:tensor [dim inter] dtype]]]
                 (fn [x gw uw]
                   (let [gate (gemma/linear x gw nil)
                         up (gemma/linear x uw nil)]
                     (swiglu gate up))))))

(defn make-swiglu-inputs
  [batch seq-len dim _dtype]
  (let [inter (* 4 dim)]
    {:x (float-array (repeat (* batch seq-len dim) 1.0))
     :gate_w (float-array (repeat (* dim inter) 0.01))
     :up_w (float-array (repeat (* dim inter) 0.01))}))

(defn build-gqa-attn-graph
  [batch seq-len num-heads num-kv-heads head-dim dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        hidden-dim q-dim]
    (trace-graph (str "gqa_attn_" batch "_" seq-len "_" num-heads "_" num-kv-heads)
                 [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                  [:q_w [:tensor [q-dim hidden-dim] dtype]]
                  [:k_w [:tensor [kv-dim hidden-dim] dtype]]
                  [:v_w [:tensor [kv-dim hidden-dim] dtype]]
                  [:o_w [:tensor [hidden-dim q-dim] dtype]]
                  [:pos [:tensor [seq-len] :i32]]]
                 (fn [x qw kw vw ow pos]
                   (first (gemma/gemma-attention x qw kw vw ow num-heads num-kv-heads pos))))))

(defn make-gqa-attn-inputs
  [batch seq-len num-heads num-kv-heads head-dim _dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        hidden-dim q-dim]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :q_w (float-array (repeat (* q-dim hidden-dim) 0.01))
     :k_w (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :v_w (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :o_w (float-array (repeat (* hidden-dim q-dim) 0.01))
     :pos (int-array (range seq-len))}))

(defn build-gpt2-block-graph
  [batch seq-len hidden-dim dtype]
  (let [inter (* 4 hidden-dim)
        attn-dim (* 3 hidden-dim)]
    (trace-graph (str "gpt2_block_" batch "_" seq-len "_" hidden-dim)
                 [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                  [:ln1_g [:tensor [hidden-dim] dtype]]
                  [:ln1_b [:tensor [hidden-dim] dtype]]
                  [:c_attn_w [:tensor [hidden-dim attn-dim] dtype]]
                  [:c_attn_b [:tensor [attn-dim] dtype]]
                  [:c_proj_w [:tensor [hidden-dim hidden-dim] dtype]]
                  [:c_proj_b [:tensor [hidden-dim] dtype]]
                  [:ln2_g [:tensor [hidden-dim] dtype]]
                  [:ln2_b [:tensor [hidden-dim] dtype]]
                  [:mlp_fc_w [:tensor [hidden-dim inter] dtype]]
                  [:mlp_fc_b [:tensor [inter] dtype]]
                  [:mlp_proj_w [:tensor [inter hidden-dim] dtype]]
                  [:mlp_proj_b [:tensor [hidden-dim] dtype]]]
                 (fn [x ln1g ln1b cw cb pw pb ln2g ln2b fcw fcb pw2 pb2]
                   (gpt2/gpt2-block x {:ln1-g ln1g :ln1-b ln1b
                                       :c-attn-w cw :c-attn-b cb
                                       :c-proj-w pw :c-proj-b pb
                                       :ln2-g ln2g :ln2-b ln2b
                                       :mlp-fc-w fcw :mlp-fc-b fcb
                                       :mlp-proj-w pw2 :mlp-proj-b pb2} 12)))))

(defn make-gpt2-block-inputs
  [batch seq-len hidden-dim _dtype]
  (let [inter (* 4 hidden-dim)
        attn-dim (* 3 hidden-dim)]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :ln1_g (float-array (repeat hidden-dim 1.0))
     :ln1_b (float-array (repeat hidden-dim 0.0))
     :c_attn_w (float-array (repeat (* hidden-dim attn-dim) 0.01))
     :c_attn_b (float-array (repeat attn-dim 0.0))
     :c_proj_w (float-array (repeat (* hidden-dim hidden-dim) 0.01))
     :c_proj_b (float-array (repeat hidden-dim 0.0))
     :ln2_g (float-array (repeat hidden-dim 1.0))
     :ln2_b (float-array (repeat hidden-dim 0.0))
     :mlp_fc_w (float-array (repeat (* hidden-dim inter) 0.01))
     :mlp_fc_b (float-array (repeat inter 0.0))
     :mlp_proj_w (float-array (repeat (* inter hidden-dim) 0.01))
     :mlp_proj_b (float-array (repeat hidden-dim 0.0))}))

(defn build-gemma4-block-graph
  [batch seq-len hidden-dim num-heads num-kv-heads head-dim pl-dim dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        mlp-dim (* 4 hidden-dim)]
    (trace-graph (str "gemma4_block_" batch "_" seq-len "_" hidden-dim)
                 [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                  [:in_ln [:tensor [hidden-dim] dtype]]
                  [:layer_scalar [:tensor [1] dtype]]
                  [:qw [:tensor [q-dim hidden-dim] dtype]]
                  [:kw [:tensor [kv-dim hidden-dim] dtype]]
                  [:vw [:tensor [kv-dim hidden-dim] dtype]]
                  [:ow [:tensor [hidden-dim q-dim] dtype]]
                  [:qn [:tensor [head-dim] dtype]]
                  [:kn [:tensor [head-dim] dtype]]
                  [:post_attn_ln [:tensor [hidden-dim] dtype]]
                  [:pre_mlp_ln [:tensor [hidden-dim] dtype]]
                  [:post_mlp_ln [:tensor [hidden-dim] dtype]]
                  [:gate_w [:tensor [mlp-dim hidden-dim] dtype]]
                  [:up_w [:tensor [mlp-dim hidden-dim] dtype]]
                  [:down_w [:tensor [hidden-dim mlp-dim] dtype]]
                  [:per_layer_gate [:tensor [pl-dim hidden-dim] dtype]]
                  [:per_layer_proj [:tensor [hidden-dim pl-dim] dtype]]
                  [:post_per_layer_norm [:tensor [hidden-dim] dtype]]
                  [:per_layer_in [:tensor [batch seq-len pl-dim] dtype]]
                  [:pos [:tensor [seq-len] :i32]]]
                 (fn [x in-ln ls qw kw vw ow qn kn post-attn pre-mlp post-mlp gw uw dw plg plp pln pl-in pos]
                   (let [weights {:input-ln-w in-ln
                                  :layer-scalar-w ls
                                  :q-w qw :k-w kw :v-w vw :o-w ow
                                  :q-norm-w qn :k-norm-w kn
                                  :post-attn-ln-w post-attn :pre-mlp-ln-w pre-mlp :post-mlp-ln-w post-mlp
                                  :gate-w gw :up-w uw :down-w dw
                                  :per-layer-gate-w plg :per-layer-proj-w plp :post-per-layer-norm-w pln
                                  :per-layer-input pl-in}]
                     (gemma/gemma-block x weights num-heads num-kv-heads pos))))))

(defn make-gemma4-block-inputs
  [batch seq-len hidden-dim num-heads num-kv-heads head-dim pl-dim _dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        mlp-dim (* 4 hidden-dim)]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :in_ln (float-array (repeat hidden-dim 1.0))
     :layer_scalar (float-array [1.0])
     :qw (float-array (repeat (* q-dim hidden-dim) 0.01))
     :kw (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :vw (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :ow (float-array (repeat (* hidden-dim q-dim) 0.01))
     :qn (float-array (repeat head-dim 1.0))
     :kn (float-array (repeat head-dim 1.0))
     :post_attn (float-array (repeat hidden-dim 1.0))
     :pre_mlp (float-array (repeat hidden-dim 1.0))
     :post_mlp (float-array (repeat hidden-dim 1.0))
     :gate_w (float-array (repeat (* mlp-dim hidden-dim) 0.01))
     :up_w (float-array (repeat (* mlp-dim hidden-dim) 0.01))
     :down_w (float-array (repeat (* hidden-dim mlp-dim) 0.01))
     :per_layer_gate (float-array (repeat (* pl-dim hidden-dim) 0.01))
     :per_layer_proj (float-array (repeat (* hidden-dim pl-dim) 0.01))
     :post_per_layer_norm (float-array (repeat hidden-dim 1.0))
     :per_layer_in (float-array (repeat (* batch seq-len pl-dim) 1.0))
     :pos (int-array (range seq-len))}))

(def WORKLOADS
  {:gemm-fp32
   {:id :gemm-fp32
    :name "GEMM FP32 (1024 x 1024 x 1024)"
    :category :gemm
    :flops (bcore/gemm-flops 1024 1024 1024)
    :bytes (* 4 3 1024 1024)
    :build-graph-fn (fn [] (build-gemm-graph 1024 1024 1024 :f32))
    :make-inputs-fn (fn [] (make-gemm-inputs 1024 1024 1024 :f32))}

   :gemm-bf16
   {:id :gemm-bf16
    :name "GEMM BF16 (1024 x 1024 x 1024)"
    :category :gemm
    :flops (bcore/gemm-flops 1024 1024 1024)
    :bytes (* 2 3 1024 1024)
    :build-graph-fn (fn [] (build-gemm-graph 1024 1024 1024 :bf16))
    :make-inputs-fn (fn [] (make-gemm-inputs 1024 1024 1024 :bf16))}

   :rms-norm
   {:id :rms-norm
    :name "RMSNorm (1 x 2048 x 4096)"
    :category :norm-act
    :flops (* 1 2048 4096 3)
    :bytes (* 4 2 2048 4096)
    :build-graph-fn (fn [] (build-rms-norm-graph 1 2048 4096 :f32))
    :make-inputs-fn (fn [] (make-rms-norm-inputs 1 2048 4096 :f32))}

   :swiglu
   {:id :swiglu
    :name "SwiGLU Activation (1 x 2048 x 4096)"
    :category :norm-act
    :flops (bcore/gemm-flops 2048 (* 4 4096) 4096)
    :bytes (* 4 4 2048 4096)
    :build-graph-fn (fn [] (build-swiglu-graph 1 2048 4096 :f32))
    :make-inputs-fn (fn [] (make-swiglu-inputs 1 2048 4096 :f32))}

   :gqa-causal-attn
   {:id :gqa-causal-attn
    :name "GQA Causal Attention (1 x 128 x 8 x 256)"
    :category :attn
    :flops (bcore/gemm-flops 128 2048 2048)
    :bytes (* 4 3 128 2048)
    :build-graph-fn (fn [] (build-gqa-attn-graph 1 128 8 1 256 :f32))
    :make-inputs-fn (fn [] (make-gqa-attn-inputs 1 128 8 1 256 :f32))}

   :gpt2-block
   {:id :gpt2-block
    :name "GPT-2 Transformer Layer Block (1 x 128 x 768)"
    :category :layer
    :flops (* 12 (bcore/gemm-flops 128 768 768))
    :bytes (* 4 10 128 768)
    :build-graph-fn (fn [] (build-gpt2-block-graph 1 128 768 :f32))
    :make-inputs-fn (fn [] (make-gpt2-block-inputs 1 128 768 :f32))}

   :gemma4-block
   {:id :gemma4-block
    :name "Gemma 4 Transformer Layer Block (1 x 128 x 1536)"
    :category :layer
    :flops (* 16 (bcore/gemm-flops 128 1536 1536))
    :bytes (* 4 12 128 1536)
    :build-graph-fn (fn [] (build-gemma4-block-graph 1 128 1536 8 1 256 256 :f32))
    :make-inputs-fn (fn [] (make-gemma4-block-inputs 1 128 1536 8 1 256 256 :f32))}})
