# Handoff to the coding workspace

Read README, TEAM_DIRECTION and EXPERIMENT_PLAN before editing. The separately supplied professor proposal is background; these documents record subsequent team decisions. Do not alter research scope or add random-routing baselines without discussing it with the team.

1. Audit the paired data and provenance, verify model configs and eligible benchmarks, inspect leakage/duplicates and costs.
2. Resolve fresh-test versus nested-evaluation design, primary selection allowance/rule, and valid OOD domains. Record prior pilot exposure honestly.
3. Implement a shared loader, split manifests, training-only preprocessing, fixed-policy evaluator and fixed-model baselines. Preserve the historical CPU pilot as development evidence.
4. Implement candidate training: logistic regression, random forest, ModernBERT GPU classification head. Confirm GPU with a small smoke test before scheduling training. Candidate head choices may differ: the existing forest uses squared-error regression; keep this documented.
5. Run the bounded development comparison and select one primary algorithm according to the fixed validation criterion.
6. Run main and holdout experiments with the selected algorithm, freezing every fitted policy before its evaluation. Save chosen predictions and metrics to reproduce reports.
7. Produce clear tables/plots, uncertainty estimates and failure analysis. Report gross API savings separately from router overhead. Keep paid API calls disabled unless explicitly authorized.

Current package includes no ModernBERT result, no GPU training implementation, no final-test claim and no selected winning algorithm. Do not infer otherwise from earlier chat recommendations.
