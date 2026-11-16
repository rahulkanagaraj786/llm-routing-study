# Current team direction

## Aim and research questions
Build a prompt-only router between the dataset's GPT-5 reasoning/medium configuration and Gemini 2.5 Flash. Verify the exact raw identifiers and configuration in provenance before labeling final plots; do not mix GPT-5-chat and GPT-5-medium records.

RQ1: How much can an LLM router save compared with always using the stronger LLM, and how does this affect answer accuracy?
RQ2: Does a router still save money while maintaining answer quality on types of questions it has not seen before?

The study concerns practical cost/accuracy trade-offs and transfer. It does not claim a novel routing algorithm, universal optimality, or superiority over untested routing strategies. A finding that an allowance fails on unfamiliar types is a valid outcome.

## Model pair
Main pair: GPT-5 and Gemini 2.5 Flash, using historical paired outcomes and costs. Sonnet is not selected. Do not assume a model is cheaper solely from its name or compare today's token prices with historical bills. Costs are recorded response costs, not payments incurred by replaying this dataset.

## Prediction and policy
For prompt i, observed Y_i = strong_correct_i - cheap_correct_i, in {-1,0,+1}. A tie includes both right and both wrong. Use only genuinely binary correctness records; an arbitrary continuous quality score is not automatically a correctness label.

Inputs are features available before generation: prompt text, with optional word count, digits, math/code indicators. No response text, scores, ground truth, output tokens, actual response costs, or benchmark identity as core prediction inputs. Benchmark/domain metadata organize splits and analysis. A standard task_category field is not assumed to exist.

For three-class predictors, gain f_hat(x) = P_hat(+1|x) - P_hat(-1|x). A regression forest can estimate average Y directly. Choose GPT-5 if gain > tau; otherwise Flash.

Training predicts correctness advantage. Validation chooses tau to minimize recorded total cost among policies whose accuracy is at least always-GPT-5 accuracy minus allowance delta. Cost is not in the current predictive training loss. This optimizes within the threshold-policy family, not all possible routing policies. Allowance delta and decision threshold tau are different quantities.

## Candidate algorithms and main study
Preliminary candidates: regularized multinomial logistic regression, random forest, fine-tuned ModernBERT. Select ONE primary algorithm on permitted validation data, then use it for the main experiments. Include the candidate comparison briefly in the report. Do not declare ModernBERT the winner because it is more complex.

One algorithm can require multiple fits. For each excluded-domain experiment, fit a separate instance without that domain and choose its cutoff without it. Fix algorithm choice and tuning protocol without consulting excluded-domain outcomes.

Primary baselines: always GPT-5 and always Flash on exactly the same eligible questions. No random-routing experiment or conclusion is planned.

## Outputs
Trained router plus preprocessing and cutoffs; cost/accuracy tables and curves; performance by benchmark; transfer results; uncertainty estimates; training and prediction overhead; reproducible code. No guarantee of cost savings or of meeting a validation allowance on test.
