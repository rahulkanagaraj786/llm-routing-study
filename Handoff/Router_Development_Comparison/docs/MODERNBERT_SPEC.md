# ModernBERT GPU candidate: implementation specification

Status: planned, not implemented or trained in this package.

Start with pretrained answerdotai/ModernBERT-base; pin a repository revision and compatible software versions when implementing. Add a three-class sequence-classification head. Map labels -1->0, 0->1, +1->2 and preserve that ordering in saved artifacts. Train encoder plus head with cross-entropy on the observed correctness-difference labels. Convert predicted class probabilities to gain p(class 2)-p(class 0).

No paid GPT-5 or Flash calls are required. ModernBERT receives prompts only; recorded model grades supply supervised labels. GPU memory requirements depend on prompt length, batch size, precision, optimizer and attention implementation; do not promise a specific GPU fit without a smoke test.

Audit token lengths before selecting a maximum sequence length. Record truncation explicitly, especially for code prompts. Start with conservative batch sizes and use supported mixed precision or gradient accumulation if needed. Train for a modest maximum number of epochs with validation-based early stopping. Keep a small search over learning rate/regularization; do not use the final test for checkpoints or threshold choice.

Optional ablation: freeze encoder and train only a head, using a documented pooling/token representation. This distinguishes encoder adaptation from merely changing input representation. A pretrained base encoder is not automatically optimized as a sentence-embedding model.

Save model/tokenizer, label map, gain computation, cutoff per allowance, config, split manifests, metrics, hardware, package versions and provenance. Implement exactly the same policy evaluator used by the CPU candidates. Main study proceeds only after validation selection; no assumed superiority.
