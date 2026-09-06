(ns clj-xla.logic.gpt2-test
  "Unit and compilation parity tests for declarative Tensor Logic GPT-2 model."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.gpt2 :as gpt2]
            [clj-xla.models.gpt2 :as gpt2-legacy]
            [clj-xla.stablehlo :as shlo]
            [clj-xla.trace :as trace]
            [clojure.test :refer [deftest is]]))

(deftest test-gpt2-single-layer-lowering
  (let [num-layers 1
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos_ids [:tensor [1 max-seq-len] :i32]]
                      [:ln_f_g [:tensor [768] :f32]]
                      [:ln_f_b [:tensor [768] :f32]]
                      [:wte [:tensor [50257 768] :f32]]
                      [:wpe [:tensor [1024 768] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                             (range num-layers)))
        ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "gpt2_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gpt2-12-layer-lowering
  (let [num-layers 12
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos_ids [:tensor [1 max-seq-len] :i32]]
                      [:ln_f_g [:tensor [768] :f32]]
                      [:ln_f_b [:tensor [768] :f32]]
                      [:wte [:tensor [50257 768] :f32]]
                      [:wpe [:tensor [1024 768] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                             (range num-layers)))
        ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "gpt2_12_layers" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gpt2-legacy-tracer-parity
  (let [num-layers 1
        max-seq-len 8
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos_ids [:tensor [1 max-seq-len] :i32]]
                      [:ln_f_g [:tensor [768] :f32]]
                      [:ln_f_b [:tensor [768] :f32]]
                      [:wte [:tensor [50257 768] :f32]]
                      [:wpe [:tensor [1024 768] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                             (range num-layers)))
        ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph-logic (lower/ast->graph "parity_logic" invars ast #{:logits})
        graph-trace (trace/trace-graph "parity_trace" invars
                                       (fn [x pos ln_f_g ln_f_b wte wpe & flat-weights]
                                         (let [layer-maps (mapv (fn [chunk]
                                                                  {:ln1-g (nth chunk 0) :ln1-b (nth chunk 1)
                                                                   :c-attn-w (nth chunk 2) :c-attn-b (nth chunk 3)
                                                                   :c-proj-w (nth chunk 4) :c-proj-b (nth chunk 5)
                                                                   :ln2-g (nth chunk 6) :ln2-b (nth chunk 7)
                                                                   :mlp-fc-w (nth chunk 8) :mlp-fc-b (nth chunk 9)
                                                                   :mlp-proj-w (nth chunk 10) :mlp-proj-b (nth chunk 11)})
                                                                (partition 12 flat-weights))]
                                           (gpt2-legacy/full-gpt2-forward x pos ln_f_g ln_f_b wte wpe layer-maps))))
        ctx (xla/get-context)
        exec-logic (xla/compile-graph ctx graph-logic)
        exec-trace (xla/compile-graph ctx graph-trace)]
    (is (some? exec-logic))
    (is (some? exec-trace))))
