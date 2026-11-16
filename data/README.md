# Data

The study uses the public LLMRouterBench release
(https://huggingface.co/datasets/NPULH/LLMRouterBench). The data is not stored in this
repository; it is rebuilt from that release.

This is a one-time step per machine. The table it produces stays in `data/processed/`
and everything else reads that local copy.

```bash
# 1. download the archive (about 1.3 GB) at the pinned revision
mkdir -p data/raw/extracted
curl -L -o data/raw/bench-release.tar.gz \
  "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz"
shasum -a 256 data/raw/bench-release.tar.gz
# expected: b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1

# 2. extract only the GPT-5 and Gemini 2.5 Flash result files
tar -xzf data/raw/bench-release.tar.gz -C data/raw/extracted -T data/selected_members.txt

# 3. build the paired table (GPT-5 and Flash outcomes joined question by question)
python scripts/01_build_paired.py
```

Step 3 writes `data/processed/paired_v2.jsonl.gz` and stops with an error if the table is
not identical to the one the study uses. After that the archive can be deleted.
