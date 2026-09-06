(ns clj-xla.logic.expand
  "Recursive subgraph expansion and N-ary contraction decomposition for Tensor Logic AST."
  (:require [clj-xla.logic.ast :as ast]
            [clojure.set :as set]
            [clojure.walk :as walk]))

(defmulti expand-node
  "Expands a high-level container or composite layer node into primitive equations."
  (fn [node _ctx]
    (when (vector? node) (first node))))

(defmethod expand-node :default [_ _] nil)

(defn- decompose-n-ary-eqn
  "Decomposes an equation with N > 2 body terms into a sequence of binary contractions."
  [eqn-node ctx]
  (let [head (ast/head eqn-node)
        head-name (first head)
        head-idxs (vec (rest head))
        attrs (ast/attrs eqn-node)
        body (ast/body-terms eqn-node)]
    (if (<= (count body) 2)
      nil ;; Already binary or unary, no decomposition needed
      (let [counter (or (:synth-counter ctx) (atom 0))
            [_first-two & remaining] (partition-all 2 body)
            ;; We fold pairwise
            steps
            (reduce
             (fn [[acc prev-term] next-term]
               (let [prev-idxs (vec (rest prev-term))
                     next-idxs (vec (rest next-term))
                     rem-indices (set (mapcat rest remaining))
                     needed-later (set/union rem-indices (set head-idxs))
                     synth-idxs (vec (filter #(contains? needed-later %)
                                             (distinct (concat prev-idxs next-idxs))))
                     synth-id (keyword (str "synth_" (swap! counter inc)))
                     synth-head (into [synth-id] synth-idxs)
                     step-eqn [:= synth-head prev-term next-term]]
                 [(conj acc step-eqn) synth-head]))
             [[] (first body)]
             (rest body))
            [acc _final-term] steps
            ;; Replace the last synthetic head with the actual head and attributes
            last-eqn (last acc)
            last-body (drop 2 last-eqn)
            final-eqn (cond-> [:= (into [head-name] head-idxs)]
                        attrs (conj attrs)
                        true (into last-body))
            all-eqns (conj (vec (butlast acc)) final-eqn)]
        all-eqns))))

(defmethod expand-node := [node ctx]
  (decompose-n-ary-eqn node ctx))

(defmethod expand-node :block [node _ctx]
  (let [children (if (map? (second node))
                   (drop 2 node)
                   (rest node))]
    (vec children)))

(defmethod expand-node :residual [node _ctx]
  ;; [:residual [:out ...] [:in ...] & children]
  ;; Automatically handles residual addition by emitting implicit accumulation or sum
  (let [elems (rest node)
        [head in-head & children] (if (map? (first elems)) (rest elems) elems)
        head-idxs (vec (rest head))
        in-head-idxs (vec (rest in-head))
        skip-eqn [:= (into [(first head)] head-idxs) (into [(first in-head)] in-head-idxs)]]
    (into [skip-eqn] children)))

(defn- statement? [node]
  (and (vector? node)
       (>= (count node) 2)
       (let [op (first node)]
         (or (keyword? op) (symbol? op)))
       (vector? (second node))))

(defn- flatten-statements [tree]
  (cond
    (nil? tree) []
    (statement? tree) [tree]
    (sequential? tree) (vec (mapcat flatten-statements tree))
    :else []))

(defn expand-ast
  "Recursively expands Hiccup AST using fixed-point prewalk until only primitive
   equations ([:= ...], direct StableHLO hooks) remain."
  ([ast] (expand-ast {} ast))
  ([ctx ast]
   (let [synth-counter (atom 0)
         full-ctx (assoc ctx :synth-counter synth-counter)
         step-fn (fn [tree]
                   (walk/prewalk
                    (fn [node]
                      (if-let [expanded (expand-node node full-ctx)]
                        expanded
                        node))
                    tree))]
     (loop [current ast
            depth 0]
       (if (> depth 100)
         (throw (ex-info "Infinite loop detected during AST expansion" {:ast ast}))
         (let [stepped (step-fn current)]
           (if (= current stepped)
             (flatten-statements stepped)
             (recur stepped (inc depth)))))))))
