# Experiment protocol

## 1. Data audit
Check raw model identifiers, matched prompt/ground truth, score semantics, missing/zero costs, duplicates and near duplicates, benchmark sizes, and per-model failures. Report exclusions and coverage. Do not silently replace missing prices with zero or recompute historical bills at current prices. Keep an explicit domain mapping with evidence. A benchmark holdout is not necessarily a domain holdout: mixed benchmarks may contain the supposedly excluded topic.

## 2. Freeze the evaluation design BEFORE new tuning
Existing repeated pilot splits exposed the available paired corpus as development data. Merely resplitting it does not restore a pristine final test set. Either obtain eligible questions never used for development, or use a clearly described nested/grouped evaluation and acknowledge earlier exploration. Do not label reused questions an untouched final confirmation. Reserve proposed OOD domains from all future algorithm selection and tuning; document prior pilot exposure as a limitation.

Fit TF-IDF vocabulary, scaling, feature selection and other learned preprocessing on training only. Group exact/near duplicates together. Split all model records for a prompt together. Maintain split manifests, hashes and seeds.

## 3. Small candidate comparison
Use the same eligible questions and splits for logistic regression, forest, and ModernBERT. Record feature differences explicitly; comparing TF-IDF to a fine-tuned encoder compares full pipelines, not only classifiers. A frozen-encoder comparison is optional.

Use a modest, declared search budget. Tune training hyperparameters only on permitted development data. If that selection is extensive, use a separate calibration subset for final threshold selection. GPU availability is supplied by the team; it is not established in this workspace.

Suggested allowances: 0, 1, 2 percentage points. These are researcher-chosen scenarios, not industry standards or guaranteed test bounds. Decide a primary allowance or a prespecified across-allowance selection rule before viewing new results. This remains an open team decision. Do not select a different winner per test result.

For each candidate and allowance, sweep validation prediction thresholds, including always-strong and always-cheap endpoints. Use strong if gain > tau. Compute chosen correctness and cost; minimize cost among feasible policies. Break ties with higher accuracy, then lower prediction overhead. Select the primary algorithm using the prespecified validation rule; record reasons.

## 4. Main study with selected algorithm
Freeze each fitted model, preprocessing and calibrated cutoff before evaluation. Score chosen outcomes from recorded data only after decisions. Report always-GPT-5 and always-Flash baselines on the identical questions. Different allowances reuse one fitted predictor with different cutoffs; they do not require retraining.

Accuracy = mean(selected correctness).
Cost = sum(selected recorded costs).
Savings percent = 100*(1 - router_cost/strong_cost).
Accuracy loss in percentage points = 100*(strong_accuracy - router_accuracy).
Negative loss means improvement. Report actual test loss even when it exceeds the allowance. The constraint is satisfied empirically on validation, not guaranteed on future requests.

Report pooled accuracy and per-benchmark results; include an equal-benchmark summary to show large datasets' influence. Quantify uncertainty with paired question/group bootstrap intervals, preserving paired outcomes and fixed policies. These intervals do not account for all model-selection uncertainty. Avoid interpreting overlapping pilot seeds as independent replications.

## 5. Transfer extension
Preselect two or three sufficiently supported holdouts after auditing domain definitions and sample size, not after choosing easy outcomes. For each, train and calibrate without the held-out content, then evaluate with the cutoff frozen. Reuse the selected algorithm and declared settings/protocol. Exclude target-domain content from mixed benchmarks if claiming unseen-domain transfer; otherwise name the experiment unseen-benchmark transfer.

## 6. Compute and reporting
Record training duration, hardware, batch size, prediction latency, truncation rate and encoder cost/overhead. Baseline recorded API savings exclude the router's own compute; label them gross savings. Do not claim net deployment savings without measuring/estimating serving costs on specified hardware and workload. No live paid LLM requests are needed for offline evaluation.

Do not promise a favorable result, change allowances after inspecting test accuracy, or claim comparison against random strategies. The final conclusion is conditional on the selected pair, recorded graders/costs, workload mix and evaluation protocol.
