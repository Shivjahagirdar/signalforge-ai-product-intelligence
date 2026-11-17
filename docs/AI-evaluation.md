# AI evaluation
Reproduce: `python scripts/benchmark.py` (seed 42; output saved in `research/benchmark-output.txt`).
On the included **synthetic** dataset: ARI 0.999, pairwise precision 1.000, recall 0.998, coverage 1.000, ~0.2 s latency.
**Interpretation:** this validates the pipeline mechanics only. Synthetic templates share anchor keywords per issue, so real, noisy text will score lower. Cluster cohesion is a clustering-quality signal, not a calibrated probability. Not yet done: noise-sensitivity sweeps, multi-seed stability, human-correction effort, human study (unmeasured; protocol to be written).
