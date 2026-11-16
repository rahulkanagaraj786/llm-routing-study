# LLM Routing: Balancing Cost and Answer Accuracy

Course project for STA 221 (Fall 2026).

We study a simple question: if a router looks at a prompt and decides whether to send it
to a stronger, more expensive LLM or a cheaper one, how much money does that save, and
what does it cost in answer accuracy?

- **RQ1.** How much can a router save compared with always using the stronger model, and
  how does that affect accuracy?
- **RQ2.** Does that trade-off hold on types of questions the router has not seen?

The study uses recorded outcomes from the public LLMRouterBench benchmark for GPT-5 and
Gemini 2.5 Flash. Nothing here calls an LLM.

## Status

Work in progress. The data build step is in place (see `data/README.md`); the rest of
the code and the results will be added as the study is finalised.

## Layout

```
src/router/   library code (data loading, routing policy, models)
scripts/      numbered scripts that run the pipeline end to end
tests/        checks for the evaluation code
docs/         protocol, data audit, results write-ups
data/         instructions for obtaining the data (the data itself is not stored here)
manifests/    train / validation / test split definitions
results/      outputs produced by the scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then follow `data/README.md` to download and build the data (one time per machine).
