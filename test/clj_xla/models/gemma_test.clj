(ns clj-xla.models.gemma-test
  "Unit and generative tests for Gemma 2 / Gemma 3 / Gemma 4 model configuration, key mapping, and tracing."
  (:require [clj-xla.models.gemma :refer [full-gemma-forward full-gemma4-forward gemma-block gemma-config gemma3-config gemma3-weight-key-map gemma4-config gemma4-weight-key-map weight-key-map]]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.gemma2-inference :as gemma2-script]))

(defspec prop-gemma-config-defaults 50
  (prop/for-all [vocab-sz (gen/choose 1000 300000)]
                (let [cfg (gemma-config {:vocab-size vocab-sz})]
                  (and (= vocab-sz (:vocab-size cfg))
                       (= 2048 (:hidden-dim cfg))
                       (= 18 (:num-layers cfg))
                       (= 8 (:num-heads cfg))
                       (= 1 (:num-kv-heads cfg))
                       (= 256 (:head-dim cfg))))))

(deftest gemma-weight-key-map-test
  (testing "Weight key map generates exact HuggingFace Gemma key names including Gemma 2 RMSNorms"
    (let [km (weight-key-map 0)]
      (is (= "model.layers.0.input_layernorm.weight" (:input-ln-w km)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (:q-w km)))
      (is (= "model.layers.0.self_attn.k_proj.weight" (:k-w km)))
      (is (= "model.layers.0.self_attn.v_proj.weight" (:v-w km)))
      (is (= "model.layers.0.self_attn.o_proj.weight" (:o-w km)))
      (is (= "model.layers.0.post_attention_layernorm.weight" (:post-attn-ln-w km)))
      (is (= "model.layers.0.pre_feedforward_layernorm.weight" (:pre-mlp-ln-w km)))
      (is (= "model.layers.0.post_feedforward_layernorm.weight" (:post-mlp-ln-w km)))
      (is (= "model.layers.0.mlp.gate_proj.weight" (:gate-w km)))
      (is (= "model.layers.0.mlp.up_proj.weight" (:up-w km)))
      (is (= "model.layers.0.mlp.down_proj.weight" (:down-w km))))))

(deftest gemma-block-tracer-test
  (testing "Gemma 2 block returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 16 2304] :f32])
          weights {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}]
      (is (tracer? (gemma-block x weights 8 4 [0]))))))

(deftest full-gemma-forward-tracer-test
  (testing "Full Gemma 2 forward pass traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}
          graph (trace-graph "gemma2_test"
                             [[:x [:tensor [1 16] :i32]]
                              [:emb [:tensor [256000 2304] :f32]]
                              [:fn_norm [:tensor [2304] :f32]]]
                             (fn [x emb fn-norm]
                               (full-gemma-forward x emb [layer-w] fn-norm [0])))]
      (is (= "gemma2_test" (:name graph)))
      (is (seq (:eqns graph)))))

  (testing "Full Gemma forward pass with kv-caches returns [logits updated-kv-caches]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 4 32 256] :f32])
                      (t/->Tracer :vc [:tensor [1 4 32 256] :f32])]]
          x (t/->Tracer :x [:tensor [1 1] :i32])
          emb (t/->Tracer :emb [:tensor [256000 2304] :f32])
          fn-norm (t/->Tracer :fn_norm [:tensor [2304] :f32])
          [logits updated-kv] (full-gemma-forward x emb [layer-w] fn-norm [0] 8 4 kv-caches 5)]
      (is (tracer? logits))
      (is (= 1 (count updated-kv)))))

  (testing "Single-Pass Prompt Prefill traces full prompt sequence [1 6]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 4 128 256] :f32])
                      (t/->Tracer :vc [:tensor [1 4 128 256] :f32])]]
          graph (trace-graph "prefill_test"
                             [[:x [:tensor [1 6] :i32]]
                              [:pos [:tensor [6] :i32]]
                              [:emb [:tensor [256000 2304] :f32]]
                              [:fn_norm [:tensor [2304] :f32]]
                              [:in_ln [:tensor [2304] :f32]]
                              [:qw [:tensor [2048 2304] :f32]]
                              [:kw [:tensor [1024 2304] :f32]]
                              [:vw [:tensor [1024 2304] :f32]]
                              [:ow [:tensor [2304 2048] :f32]]
                              [:post_attn_ln [:tensor [2304] :f32]]
                              [:pre_mlp_ln [:tensor [2304] :f32]]
                              [:post_mlp_ln [:tensor [2304] :f32]]
                              [:gw [:tensor [9216 2304] :f32]]
                              [:uw [:tensor [9216 2304] :f32]]
                              [:dw [:tensor [2304 9216] :f32]]
                              [:kc [:tensor [1 4 128 256] :f32]]
                              [:vc [:tensor [1 4 128 256] :f32]]]
                             (fn [x pos emb fn_norm & _rest]
                               (let [[logits updated-kv] (full-gemma-forward x emb [layer-w] fn_norm pos 8 4 kv-caches 0)]
                                 (into [logits] (apply concat updated-kv)))))]
      (is (= "prefill_test" (:name graph)))
      (is (seq (:eqns graph)))))

  (testing "BF16 precision forward pass traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :bf16])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :bf16])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :bf16])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :bf16])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :bf16])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :bf16])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :bf16])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :bf16])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :bf16])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :bf16])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :bf16])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 4 128 256] :bf16])
                      (t/->Tracer :vc [:tensor [1 4 128 256] :bf16])]]
          x (t/->Tracer :x [:tensor [1 1] :i32])
          emb (t/->Tracer :emb [:tensor [256000 2304] :bf16])
          fn-norm (t/->Tracer :fn_norm [:tensor [2304] :bf16])
          [logits updated-kv] (full-gemma-forward x emb [layer-w] fn-norm [0] 8 4 kv-caches 0)
          f32-logits (t/convert logits :f32)]
      (is (tracer? logits))
      (is (= [:tensor [1 1 256000] :f32] (:type f32-logits)))
      (is (= 1 (count updated-kv))))))

(deftest gemma2-cli-args-test
  (testing "parse-cli-args parses --verbose flag correctly"
    (let [opts-default (gemma2-script/parse-cli-args [])
          opts-verbose (gemma2-script/parse-cli-args ["--verbose"])
          opts-combined (gemma2-script/parse-cli-args ["--prompt" "Hello" "--verbose"])]
      (is (false? (:verbose opts-default)))
      (is (true? (:verbose opts-verbose)))
      (is (true? (:verbose opts-combined)))
      (is (= "Hello" (:prompt opts-combined))))))

(deftest gemma2-find-model-dir-test
  (testing "find-model-dir throws ex-info when safetensors file does not exist in any model-dir"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Model directory with safetensors not found"
                          (gemma2-script/find-model-dir ["/non/existent/path/1" "/non/existent/path/2"])))))

(defspec prop-gemma3-config-defaults 50
  (prop/for-all [vocab-sz (gen/choose 1000 300000)]
                (let [cfg (gemma3-config {:vocab-size vocab-sz})]
                  (and (= vocab-sz (:vocab-size cfg))
                       (= 640 (:hidden-dim cfg))
                       (= 2048 (:intermediate-dim cfg))
                       (= 18 (:num-layers cfg))
                       (= 4 (:num-heads cfg))
                       (= 1 (:num-kv-heads cfg))
                       (= 160 (:head-dim cfg))))))

(deftest gemma3-weight-key-map-test
  (testing "gemma3-weight-key-map generates exact HuggingFace Gemma 3 key names including QK norms"
    (let [km (gemma3-weight-key-map 0)]
      (is (= "model.layers.0.input_layernorm.weight" (:input-ln-w km)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (:q-w km)))
      (is (= "model.layers.0.self_attn.k_proj.weight" (:k-w km)))
      (is (= "model.layers.0.self_attn.v_proj.weight" (:v-w km)))
      (is (= "model.layers.0.self_attn.o_proj.weight" (:o-w km)))
      (is (= "model.layers.0.self_attn.q_norm.weight" (:q-norm-w km)))
      (is (= "model.layers.0.self_attn.k_norm.weight" (:k-norm-w km)))
      (is (= "model.layers.0.post_attention_layernorm.weight" (:post-attn-ln-w km)))
      (is (= "model.layers.0.pre_feedforward_layernorm.weight" (:pre-mlp-ln-w km)))
      (is (= "model.layers.0.post_feedforward_layernorm.weight" (:post-mlp-ln-w km)))
      (is (= "model.layers.0.mlp.gate_proj.weight" (:gate-w km)))
      (is (= "model.layers.0.mlp.up_proj.weight" (:up-w km)))
      (is (= "model.layers.0.mlp.down_proj.weight" (:down-w km))))))

(defspec prop-gemma4-config-defaults 50
  (prop/for-all [vocab-sz (gen/choose 1000 300000)]
                (let [cfg (gemma4-config {:vocab-size vocab-sz})]
                  (and (= vocab-sz (:vocab-size cfg))
                       (= 1536 (:hidden-dim cfg))
                       (= 6144 (:intermediate-dim cfg))
                       (= 35 (:num-layers cfg))
                       (= 8 (:num-heads cfg))
                       (= 1 (:num-kv-heads cfg))
                       (= 256 (:head-dim cfg))
                       (= 256 (:pl-dim cfg))))))

(defspec prop-gemma4-variant-configs 50
  (prop/for-all [variant (gen/elements [:e2b :e4b :12b])]
                (let [cfg (gemma4-config variant)
                      [exp-hidden exp-inter exp-layers exp-heads exp-kv-heads exp-pl exp-shared]
                      (case variant
                        :e2b [1536 6144 35 8 1 256 20]
                        :e4b [2560 10240 42 8 2 256 18]
                        :12b [3840 15360 48 16 8 0 0])]
                  (and (= exp-hidden (:hidden-dim cfg))
                       (= exp-inter (:intermediate-dim cfg))
                       (= exp-layers (:num-layers cfg))
                       (= exp-heads (:num-heads cfg))
                       (= exp-kv-heads (:num-kv-heads cfg))
                       (= 256 (:head-dim cfg))
                       (= exp-pl (:pl-dim cfg))
                       (= exp-shared (:num-kv-shared-layers cfg))))))

(deftest gemma4-weight-key-map-test
  (testing "gemma4-weight-key-map generates exact HuggingFace Gemma 4 key names including per-layer input keys"
    (let [km (gemma4-weight-key-map 0)]
      (is (= "model.layers.0.input_layernorm.weight" (:input-ln-w km)))
      (is (= "model.layers.0.layer_scalar" (:layer-scalar-w km)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (:q-w km)))
      (is (= "model.layers.0.self_attn.k_proj.weight" (:k-w km)))
      (is (= "model.layers.0.self_attn.v_proj.weight" (:v-w km)))
      (is (= "model.layers.0.self_attn.o_proj.weight" (:o-w km)))
      (is (= "model.layers.0.self_attn.q_norm.weight" (:q-norm-w km)))
      (is (= "model.layers.0.self_attn.k_norm.weight" (:k-norm-w km)))
      (is (= "model.layers.0.post_attention_layernorm.weight" (:post-attn-ln-w km)))
      (is (= "model.layers.0.pre_feedforward_layernorm.weight" (:pre-mlp-ln-w km)))
      (is (= "model.layers.0.post_feedforward_layernorm.weight" (:post-mlp-ln-w km)))
      (is (= "model.layers.0.mlp.gate_proj.weight" (:gate-w km)))
      (is (= "model.layers.0.mlp.up_proj.weight" (:up-w km)))
      (is (= "model.layers.0.mlp.down_proj.weight" (:down-w km)))
      (is (= "model.layers.0.per_layer_input_gate.weight" (:per-layer-gate-w km)))
      (is (= "model.layers.0.per_layer_projection.weight" (:per-layer-proj-w km)))
      (is (= "model.layers.0.post_per_layer_input_norm.weight" (:post-per-layer-norm-w km))))))

(deftest full-gemma4-forward-tracer-test
  (testing "Full Gemma 4 forward pass traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [1536] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 1536] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 1536] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 1536] :f32])
                   :o-w (t/->Tracer :ow [:tensor [1536 2048] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [1536] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [1536] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [1536] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [6144 1536] :f32])
                   :up-w (t/->Tracer :uw [:tensor [6144 1536] :f32])
                   :down-w (t/->Tracer :dw [:tensor [1536 6144] :f32])
                   :per-layer-gate-w (t/->Tracer :plg [:tensor [256 1536] :f32])
                   :per-layer-proj-w (t/->Tracer :plp [:tensor [1536 256] :f32])
                   :post-per-layer-norm-w (t/->Tracer :pln [:tensor [1536] :f32])}
          graph (trace-graph "gemma4_test"
                             [[:x [:tensor [1 4] :i32]]
                              [:emb [:tensor [262144 1536] :f32]]
                              [:emb_pl [:tensor [262144 256] :f32]]
                              [:pl_proj [:tensor [256 1536] :f32]]
                              [:pl_norm [:tensor [256] :f32]]
                              [:fn_norm [:tensor [1536] :f32]]]
                             (fn [x emb emb_pl pl_proj pl_norm fn_norm]
                               (full-gemma4-forward x emb emb_pl pl_proj pl_norm [layer-w] fn_norm [0 1 2 3])))]
      (is (= "gemma4_test" (:name graph)))
      (is (seq (:eqns graph))))))

(deftest full-gemma4-e4b-forward-tracer-test
  (testing "Full Gemma 4 E4B forward pass (2 KV heads, 2560 hidden-dim) traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2560] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2560] :f32])
                   :k-w (t/->Tracer :kw [:tensor [512 2560] :f32])
                   :v-w (t/->Tracer :vw [:tensor [512 2560] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2560 2048] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2560] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2560] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2560] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [10240 2560] :f32])
                   :up-w (t/->Tracer :uw [:tensor [10240 2560] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2560 10240] :f32])
                   :per-layer-gate-w (t/->Tracer :plg [:tensor [256 2560] :f32])
                   :per-layer-proj-w (t/->Tracer :plp [:tensor [2560 256] :f32])
                   :post-per-layer-norm-w (t/->Tracer :pln [:tensor [2560] :f32])}
          graph (trace-graph "gemma4_e4b_test"
                             [[:x [:tensor [1 4] :i32]]
                              [:emb [:tensor [262144 2560] :f32]]
                              [:emb_pl [:tensor [262144 256] :f32]]
                              [:pl_proj [:tensor [256 2560] :f32]]
                              [:pl_norm [:tensor [256] :f32]]
                              [:fn_norm [:tensor [2560] :f32]]]
                             (fn [x emb emb_pl pl_proj pl_norm fn_norm]
                               (full-gemma4-forward x emb emb_pl pl_proj pl_norm [layer-w] fn_norm [0 1 2 3] 8 2)))]
      (is (= "gemma4_e4b_test" (:name graph)))
      (is (seq (:eqns graph))))))

(deftest full-gemma4-12b-forward-tracer-test
  (testing "Full Gemma 4 12B forward pass without PLE (16 heads, 8 kv heads, 3840 hidden-dim) traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [3840] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [4096 3840] :f32])
                   :k-w (t/->Tracer :kw [:tensor [2048 3840] :f32])
                   :v-w (t/->Tracer :vw [:tensor [2048 3840] :f32])
                   :o-w (t/->Tracer :ow [:tensor [3840 4096] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [3840] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [3840] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [3840] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [15360 3840] :f32])
                   :up-w (t/->Tracer :uw [:tensor [15360 3840] :f32])
                   :down-w (t/->Tracer :dw [:tensor [3840 15360] :f32])}
          graph (trace-graph "gemma4_12b_test"
                             [[:x [:tensor [1 4] :i32]]
                              [:emb [:tensor [262144 3840] :f32]]
                              [:emb_pl [:tensor [262144 0] :f32]]
                              [:pl_proj [:tensor [0 3840] :f32]]
                              [:pl_norm [:tensor [0] :f32]]
                              [:fn_norm [:tensor [3840] :f32]]]
                             (fn [x emb emb_pl pl_proj pl_norm fn_norm]
                               (full-gemma4-forward x emb emb_pl pl_proj pl_norm [layer-w] fn_norm [0 1 2 3] 16 8)))]
      (is (= "gemma4_12b_test" (:name graph)))
      (is (seq (:eqns graph))))))

(deftest gemma4-slice-last-token-test
  (testing "Gemma 4 forward pass with :slice-last-token? returns sliced logits shape [1 1 262144]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2048] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2048] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 2048] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 2048] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2048 2048] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2048] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2048] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2048] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [8192 2048] :f32])
                   :up-w (t/->Tracer :uw [:tensor [8192 2048] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2048 8192] :f32])
                   :per-layer-gate-w (t/->Tracer :plg [:tensor [256 2048] :f32])
                   :per-layer-proj-w (t/->Tracer :plp [:tensor [2048 256] :f32])
                   :post-per-layer-norm-w (t/->Tracer :pln [:tensor [2048] :f32])}
          graph (trace-graph "gemma4_slice_test"
                             [[:x [:tensor [1 128] :i32]]
                              [:emb [:tensor [262144 2048] :f32]]
                              [:emb_pl [:tensor [262144 256] :f32]]
                              [:pl_proj [:tensor [256 2048] :f32]]
                              [:pl_norm [:tensor [256] :f32]]
                              [:fn_norm [:tensor [2048] :f32]]]
                             (fn [x emb emb_pl pl_proj pl_norm fn_norm]
                               (full-gemma4-forward x emb emb_pl pl_proj pl_norm [layer-w] fn_norm [0 1 2 3] 8 1 nil nil {:slice-last-token? true})))]
      (is (= "gemma4_slice_test" (:name graph)))
      (is (seq (:eqns graph))))))

(deftest gemma4-kv-cache-forward-test
  (testing "Gemma 4 forward pass with kv-caches returns [logits updated-kv-caches]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2048] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2048] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 2048] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 2048] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2048 2048] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2048] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2048] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2048] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [8192 2048] :f32])
                   :up-w (t/->Tracer :uw [:tensor [8192 2048] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2048 8192] :f32])
                   :per-layer-gate-w (t/->Tracer :plg [:tensor [256 2048] :f32])
                   :per-layer-proj-w (t/->Tracer :plp [:tensor [2048 256] :f32])
                   :post-per-layer-norm-w (t/->Tracer :pln [:tensor [2048] :f32])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 1 128 256] :f32])
                      (t/->Tracer :vc [:tensor [1 1 128 256] :f32])]]
          x (t/->Tracer :x [:tensor [1 1] :i32])
          emb (t/->Tracer :emb [:tensor [262144 2048] :f32])
          emb-pl (t/->Tracer :emb_pl [:tensor [262144 256] :f32])
          pl-proj (t/->Tracer :pl_proj [:tensor [256 2048] :f32])
          pl-norm (t/->Tracer :pl_norm [:tensor [256] :f32])
          fn-norm (t/->Tracer :fn_norm [:tensor [2048] :f32])
          pos-tracer (t/->Tracer :pos [:tensor [1] :i32])
          [logits updated-kv] (full-gemma4-forward x emb emb-pl pl-proj pl-norm [layer-w] fn-norm pos-tracer 8 1 kv-caches 5)]
      (is (tracer? logits))
      (is (= 1 (count updated-kv))))))

(deftest gemma4-dual-graph-prefill-test
  (testing "Gemma 4 prefill graph with prompt length 16 and :slice-last-token? true returns [logits updated-kv-caches]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2048] :f32])
                   :layer-scalar-w (t/->Tracer :ls [:tensor [1] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2048] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 2048] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 2048] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2048 2048] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2048] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2048] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2048] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [8192 2048] :f32])
                   :up-w (t/->Tracer :uw [:tensor [8192 2048] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2048 8192] :f32])
                   :per-layer-gate-w (t/->Tracer :plg [:tensor [256 2048] :f32])
                   :per-layer-proj-w (t/->Tracer :plp [:tensor [2048 256] :f32])
                   :post-per-layer-norm-w (t/->Tracer :pln [:tensor [2048] :f32])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 1 128 256] :f32])
                      (t/->Tracer :vc [:tensor [1 1 128 256] :f32])]]
          x (t/->Tracer :x [:tensor [1 16] :i32])
          emb (t/->Tracer :emb [:tensor [262144 2048] :f32])
          emb-pl (t/->Tracer :emb_pl [:tensor [262144 256] :f32])
          pl-proj (t/->Tracer :pl_proj [:tensor [256 2048] :f32])
          pl-norm (t/->Tracer :pl_norm [:tensor [256] :f32])
          fn-norm (t/->Tracer :fn_norm [:tensor [2048] :f32])
          pos-tracer (t/->Tracer :pos [:tensor [16] :i32])
          [logits updated-kv] (full-gemma4-forward x emb emb-pl pl-proj pl-norm [layer-w] fn-norm pos-tracer 8 1 kv-caches 0 {:slice-last-token? true})]
      (is (tracer? logits))
      (is (= 1 (count updated-kv))))))




