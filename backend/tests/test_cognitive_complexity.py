"""
Pytest suite for cognitive complexity validator.

Ensures that changes to compute_cognitive_complexity() in code_metrics.py
immediately break CI if the SonarSource spec compliance drops.

Reference: evaluation/cognitive_complexity_validator.py BUILTIN_BENCHMARK
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluation.cognitive_complexity_validator import BUILTIN_BENCHMARK
from features.code_metrics import compute_cognitive_complexity


# ---------------------------------------------------------------------------
# Per-benchmark-case parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,source,expected",
    [
        (case["function_name"], case["source"], case["sonarqube_cc"])
        for case in BUILTIN_BENCHMARK
    ],
)
def test_cognitive_complexity_benchmark(name, source, expected):
    """Each test case must match the SonarQube specification exactly (±0 tolerance)."""
    computed = compute_cognitive_complexity(source)
    assert computed == expected, (
        f"Benchmark case '{name}': expected CC={expected}, got CC={computed}.\n"
        f"This means compute_cognitive_complexity() diverges from the SonarSource spec."
    )


# ---------------------------------------------------------------------------
# Validator-level threshold test
# ---------------------------------------------------------------------------

def test_validator_passes_builtin_benchmark():
    from evaluation.cognitive_complexity_validator import CognitiveComplexityValidator

    validator = CognitiveComplexityValidator()
    result = validator.validate(BUILTIN_BENCHMARK)
    assert result.passes_threshold(min_pearson=0.95, max_mae=2.0), (
        f"Validator failed threshold: pearson={result.pearson_r:.4f}, mae={result.mae:.4f}"
    )
