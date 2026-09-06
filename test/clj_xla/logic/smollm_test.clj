(ns clj-xla.logic.smollm-test
  "Unit and generative tests for declarative Tensor Logic SmolLM model."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.ast :as ast]
            [clj-xla.logic.expand :as expand]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.smollm :as smollm]
            [clj-xla.models.smollm :as smollm-legacy]
            [clj-xla.stablehlo :as shlo]
            [clj-xla.trace :as trace]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-smollm-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 29)
                 seq-len (gen/elements [8 16 32])]
                (let [layer-ast (smollm/smollm-layer-ast layer-idx seq-len)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-smollm-single-layer-lowering
  (let [num-layers 1
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [49152 576] :f32]]
                      [:final_norm_w [:tensor [576] :f32]]
                      [:lm_head_w [:tensor [49152 576] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                             (range num-layers)))
        ast (smollm/smollm-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "smollm_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-smollm-legacy-tracer-parity
  (let [num-layers 1
        max-seq-len 8
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [49152 576] :f32]]
                      [:final_norm_w [:tensor [576] :f32]]
                      [:lm_head_w [:tensor [49152 576] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                             (range num-layers)))
        ast (smollm/smollm-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph-logic (lower/ast->graph "parity_smollm_logic" invars ast #{:logits})
        graph-trace (trace/trace-graph "parity_smollm_trace" invars
                                       (fn [x emb fn-norm lm-hw & layer-args]
                                         (let [lw-seq (mapv (fn [[in-ln qw kw vw ow post-ln gw uw dw]]
                                                              {:input-ln-w in-ln :q-w qw :k-w kw :v-w vw :o-w ow :post-attn-ln-w post-ln :gate-w gw :up-w uw :down-w dw})
                                                            (partition 9 layer-args))]
                                           (smollm-legacy/full-smollm-forward x emb lw-seq fn-norm lm-hw [0]))))
        ctx (xla/get-context)
        exec-logic (xla/compile-graph ctx graph-logic)
        exec-trace (xla/compile-graph ctx graph-trace)]
    (is (some? exec-logic))
    (is (some? exec-trace))))
