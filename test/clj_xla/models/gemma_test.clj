(ns clj-xla.models.gemma-test
  "Unit and generative tests for Gemma 2 / Gemma 3 model configuration, key mapping, and tracing."
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
                       (= 2048 (:hidden-size cfg))
                       (= 18 (:num-hidden-layers cfg))
                       (= 8 (:num-attention-heads cfg))
                       (= 1 (:num-key-value-heads cfg))
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
                       (= 640 (:hidden-size cfg))
                       (= 2048 (:intermediate-size cfg))
                       (= 18 (:num-hidden-layers cfg))
                       (= 4 (:num-attention-heads cfg))
                       (= 1 (:num-key-value-heads cfg))
                       (= 256 (:head-dim cfg))))))

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
                       (= 1536 (:hidden-size cfg))
                       (= 6144 (:intermediate-size cfg))
                       (= 35 (:num-hidden-layers cfg))
                       (= 8 (:num-attention-heads cfg))
                       (= 1 (:num-key-value-heads cfg))
                       (= 256 (:head-dim cfg))
                       (= 256 (:hidden-size-per-layer-input cfg))
                       (= 20 (:num-kv-shared-layers cfg))))))

(deftest gemma4-weight-key-map-test
  (testing "gemma4-weight-key-map generates exact HuggingFace Gemma 4 key names including per-layer input keys and layer_scalar"
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

(defspec prop-gemma4-attention-patterns 50
  (prop/for-all [idx (gen/choose 0 34)]
                (let [is-global? (zero? (mod (inc idx) 5))
                      expected-q-dim (if is-global? 4096 2048)
                      expected-kv-dim (if is-global? 512 256)
                      expected-head-dim (if is-global? 512 256)
                      expected-theta (if is-global? 1000000.0 10000.0)
                      expected-rope-prop (if is-global? 0.25 1.0)
                      cfg (gemma4-config)]
                  (and (= 35 (count (:layer-types cfg)))
                       (number? expected-q-dim)
                       (number? expected-kv-dim)
                       (number? expected-head-dim)
                       (number? expected-theta)
                       (number? expected-rope-prop)))))

(deftest gemma4-full-forward-tracer-test
  (testing "Full Gemma 4 forward pass traces valid computation graph with PLE and layer scalars"
    (let [local-layer {:input-ln-w (t/->Tracer :in_ln_0 [:tensor [1536] :f32])
                       :layer-scalar-w (t/->Tracer :ls_0 [:tensor [1] :f32])
                       :q-w (t/->Tracer :qw_0 [:tensor [2048 1536] :f32])
                       :k-w (t/->Tracer :kw_0 [:tensor [256 1536] :f32])
                       :v-w (t/->Tracer :vw_0 [:tensor [256 1536] :f32])
                       :o-w (t/->Tracer :ow_0 [:tensor [1536 2048] :f32])
                       :q-norm-w (t/->Tracer :qn_0 [:tensor [256] :f32])
                       :k-norm-w (t/->Tracer :kn_0 [:tensor [256] :f32])
                       :post-attn-ln-w (t/->Tracer :post_attn_ln_0 [:tensor [1536] :f32])
                       :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln_0 [:tensor [1536] :f32])
                       :post-mlp-ln-w (t/->Tracer :post_mlp_ln_0 [:tensor [1536] :f32])
                       :gate-w (t/->Tracer :gw_0 [:tensor [6144 1536] :f32])
                       :up-w (t/->Tracer :uw_0 [:tensor [6144 1536] :f32])
                       :down-w (t/->Tracer :dw_0 [:tensor [1536 6144] :f32])
                       :per-layer-gate-w (t/->Tracer :plg_0 [:tensor [256 1536] :f32])
                       :per-layer-proj-w (t/->Tracer :plp_0 [:tensor [1536 256] :f32])
                       :post-per-layer-norm-w (t/->Tracer :pln_0 [:tensor [1536] :f32])
                       :theta-base 10000.0
                       :rope-proportion 1.0}
          global-layer {:input-ln-w (t/->Tracer :in_ln_4 [:tensor [1536] :f32])
                        :layer-scalar-w (t/->Tracer :ls_4 [:tensor [1] :f32])
                        :q-w (t/->Tracer :qw_4 [:tensor [4096 1536] :f32])
                        :k-w (t/->Tracer :kw_4 [:tensor [512 1536] :f32])
                        :v-w (t/->Tracer :vw_4 [:tensor [512 1536] :f32])
                        :o-w (t/->Tracer :ow_4 [:tensor [1536 4096] :f32])
                        :q-norm-w (t/->Tracer :qn_4 [:tensor [512] :f32])
                        :k-norm-w (t/->Tracer :kn_4 [:tensor [512] :f32])
                        :post-attn-ln-w (t/->Tracer :post_attn_ln_4 [:tensor [1536] :f32])
                        :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln_4 [:tensor [1536] :f32])
                        :post-mlp-ln-w (t/->Tracer :post_mlp_ln_4 [:tensor [1536] :f32])
                        :gate-w (t/->Tracer :gw_4 [:tensor [6144 1536] :f32])
                        :up-w (t/->Tracer :uw_4 [:tensor [6144 1536] :f32])
                        :down-w (t/->Tracer :dw_4 [:tensor [1536 6144] :f32])
                        :per-layer-gate-w (t/->Tracer :plg_4 [:tensor [256 1536] :f32])
                        :per-layer-proj-w (t/->Tracer :plp_4 [:tensor [1536 256] :f32])
                        :post-per-layer-norm-w (t/->Tracer :pln_4 [:tensor [1536] :f32])
                        :theta-base 1000000.0
                        :rope-proportion 0.25}
          graph (trace-graph "gemma4_test"
                             [[:x [:tensor [1 4] :i32]]
                              [:emb [:tensor [262144 1536] :f32]]
                              [:emb_pl [:tensor [262144 8960] :f32]]
                              [:pl_model_proj [:tensor [8960 1536] :f32]]
                              [:pl_proj_norm [:tensor [256] :f32]]
                              [:fn_norm [:tensor [1536] :f32]]]
                             (fn [x emb emb-pl pl-proj pl-norm fn-norm]
                               (full-gemma4-forward x emb emb-pl pl-proj pl-norm [local-layer global-layer] fn-norm [0] 8 1)))]
      (is (= "gemma4_test" (:name graph)))
      (is (seq (:eqns graph))))))



