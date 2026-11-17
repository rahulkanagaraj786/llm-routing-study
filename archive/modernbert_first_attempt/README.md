# ModernBERT, first attempt (not part of the pipeline)

An earlier, working ModernBERT router and its helper scripts, kept for reference only. It
was run once on the unfiltered pool on an Apple-silicon Mac (results in
`results/dev_selection_v1_unfiltered/modernbert/`).

The encoder model is now written by its owner against the contract in
`src/router/base.py`. Nothing in the pipeline imports the files in this folder. They may be
used as a starting point; they would need the import paths adjusted.
