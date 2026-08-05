(ns clj-xla.models.gemma-test
  "Unit and generative tests for Gemma 2 / Gemma 3 model configuration, key mapping, and tracing."
  (:require [clj-xla.models.gemma :refer [full-gemma-forward gemma-block gemma-config gemma3-config gemma3-weight-key-map weight-key-map]]
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

(deftest gemma3-block-tracer-test
  (testing "Gemma 3 block with QK norm and theta-base returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 16 640] :f32])
          weights {:input-ln-w (t/->Tracer :in_ln [:tensor [640] :f32])
                   :q-w (t/->Tracer :qw [:tensor [1024 640] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 640] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 640] :f32])
                   :o-w (t/->Tracer :ow [:tensor [640 1024] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [640] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [640] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [640] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [2048 640] :f32])
                   :up-w (t/->Tracer :uw [:tensor [2048 640] :f32])
                   :down-w (t/->Tracer :dw [:tensor [640 2048] :f32])
                   :theta-base 10000.0
                   :attn-softcap nil}]
      (is (tracer? (gemma-block x weights 4 1 [0]))))))

(deftest gemma3-full-forward-tracer-test
  (testing "Full Gemma 3 forward pass traces valid graph with QK norm and disabled softcapping"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [640] :f32])
                   :q-w (t/->Tracer :qw [:tensor [1024 640] :f32])
                   :k-w (t/->Tracer :kw [:tensor [256 640] :f32])
                   :v-w (t/->Tracer :vw [:tensor [256 640] :f32])
                   :o-w (t/->Tracer :ow [:tensor [640 1024] :f32])
                   :q-norm-w (t/->Tracer :qn [:tensor [256] :f32])
                   :k-norm-w (t/->Tracer :kn [:tensor [256] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [640] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [640] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [640] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [2048 640] :f32])
                   :up-w (t/->Tracer :uw [:tensor [2048 640] :f32])
                   :down-w (t/->Tracer :dw [:tensor [640 2048] :f32])
                   :theta-base 1000000.0
                   :attn-softcap nil}
          kv-caches [[(t/->Tracer :kc [:tensor [1 1 128 256] :f32])
                      (t/->Tracer :vc [:tensor [1 1 128 256] :f32])]]
          graph (trace-graph "gemma3_test"
                             [[:x [:tensor [1 6] :i32]]
                              [:pos [:tensor [6] :i32]]
                              [:emb [:tensor [262144 640] :f32]]
                              [:fn_norm [:tensor [640] :f32]]]
                             (fn [x _pos emb fn_norm]
                               (let [[logits updated-kv] (full-gemma-forward x emb [layer-w] fn_norm [0] 4 1 kv-caches 0 {:final-logit-softcap nil})]
                                 (into [logits] (apply concat updated-kv)))))]
      (is (= "gemma3_test" (:name graph)))
      (is (seq (:eqns graph))))))


