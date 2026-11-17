import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os  # noqa: E402

# Unit tests run offline and deterministically on the lexical backend; the semantic backend is measured by
# scripts/evaluate.py (CI) and tests/benchmarks/test_semantic.py (skipped when the model can't be downloaded).
os.environ.setdefault("SF_SEMANTIC", "0")
