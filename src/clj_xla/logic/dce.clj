(ns clj-xla.logic.dce
  "Backward-chaining dead-code elimination and implicit accumulation grouping for Tensor Logic."
  (:require [clj-xla.logic.ast :as ast]))

(defn- eqn-head-name [eqn]
  (first (ast/head eqn)))

(defn- eqn-body-names [eqn]
  (set (map first (ast/body-terms eqn))))

(defn prune-ast
  "Performs backward-chaining dead-code elimination starting from `target-heads`.
   Prunes equations that do not contribute transitively to any target head."
  [equations target-heads]
  (let [target-set (set target-heads)
        ;; Map head names to all defining equations
        head->eqns (group-by eqn-head-name equations)]
    (loop [worklist (into [] target-set)
           needed-heads #{}
           visited-heads #{}]
      (if (empty? worklist)
        ;; Filter equations in original order that produce needed heads
        (filterv (fn [eqn]
                   (contains? needed-heads (eqn-head-name eqn)))
                 equations)
        (let [curr-head (first worklist)
              rest-worklist (subvec worklist 1)]
          (if (contains? visited-heads curr-head)
            (recur rest-worklist needed-heads visited-heads)
            (let [def-eqns (get head->eqns curr-head [])
                  input-names (mapcat eqn-body-names def-eqns)
                  new-work (filter #(and (contains? head->eqns %)
                                         (not (contains? visited-heads %)))
                                   input-names)]
              (recur (into rest-worklist new-work)
                     (conj needed-heads curr-head)
                     (conj visited-heads curr-head)))))))))

(defn group-accumulations
  "Groups equations sharing an identical head signature for implicit addition lowering."
  [equations]
  (let [by-head (group-by ast/head equations)]
    (mapv (fn [eqn]
            (let [h (ast/head eqn)
                  eqns (get by-head h)]
              (if (> (count eqns) 1)
                (assoc eqn :accum-group eqns)
                eqn)))
          equations)))
