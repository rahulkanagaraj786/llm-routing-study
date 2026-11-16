# Historical CPU pilot

The run used 10,367 eligible matched questions and three overlapping 75/25 group splits, seeds 17,29,43 (7,775 training and 2,592 validation each). Predictors used TF-IDF only. Logistic regression: C=1, max_iter=250. Forest regressor: 160 trees, max_depth=20, min_samples_leaf=10, max_features=0.1. No hyperparameter search was performed.

Average validation savings versus GPT-5:

| Allowed loss (percentage points) | Logistic regression | Random forest |
|---|---:|---:|
| 0 | 0.19% | 0.16% |
| 1 | 4.72% | 3.44% |
| 2 | 9.00% | 7.78% |

Thresholds were selected and reported on the same validation sets. These are development estimates, not independent final test results or a guarantee of future savings. The available corpus has been exposed to pilot analysis. The inclusion filter is broader than the earlier six-benchmark walkthrough, so figures are not directly comparable. Random forest's weaker average here is not an exhaustive comparison, and ModernBERT has not been evaluated.

The original data source is preserved in data/source_provenance.json. The old nested archive was replaced with the exact paired data and provenance to avoid distributing obsolete study code. The out-of-scope diagnostic was removed from code and JSON; remaining historical numeric results were preserved unchanged. New reruns write a separate file. Runtime figures are hardware-specific and exclude some preprocessing work.
