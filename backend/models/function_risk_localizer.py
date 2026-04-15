"""
Function-Level Risk Localizer (FLRL)
======================================
Decomposes a file-level bug prediction into per-function risk scores using
per-function static feature extraction and complexity-weighted aggregation.

Novel contribution
------------------
All existing bug prediction models in IntelliCode (and in the literature:
Kamei 2013, Hassan 2009) operate at file or commit level.  FLRL introduces
function-granularity risk attribution without requiring new labeled data:

  1. Extract static features for each function individually (AST slice)
  2. Score each function using the trained bug predictor's heuristic path
     (which generalises better than the XGBoost path for small snippets)
  3. Weight each function's score by its share of the file's total
     cognitive complexity — more complex functions bear more responsibility
  4. Produce a ranked list of high-risk functions with:
       - raw_score        : bug predictor output for the function alone
       - complexity_weight: fraction of file's total cognitive complexity
       - attributed_score : raw_score * complexity_weight (the localised signal)
       - rank             : 1 = highest risk
       - reason           : human-readable explanation

This is different from SHAP feature attribution (which attributes a score to
input features, not to code locations) and different from complexity highlighting
(which ranks by complexity, not predicted bug probability).

The combination of predicted bug signal AND complexity weighting produces a
ranking that is more actionable than either alone:
  - A function with raw_score=0.9 but complexity_weight=0.02 ranks lower than
    one with raw_score=0.7 but complexity_weight=0.35, because the latter
    accounts for more of the file's structural risk.

Architecture
------------
No learned weights beyond the base bug predictor. FLRL is a deterministic
post-processing layer — it requires no training data and no additional
labeling. This makes it deployable immediately against any existing checkpoint.

Output
------
FunctionRiskResult(
    functions: list[FunctionRisk]   — sorted by attributed_score descending
    top_k: list[FunctionRisk]       — top-3 highest-risk functions
    file_risk_score: float          — complexity-weighted aggregate (0-1)
    total_functions: int
    n_high_risk: int                — functions with attributed_score > 0.3
)
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HEURISTIC_THRESHOLDS = {
    "low":      0.18,
    "medium":   0.40,
    "high":     0.62,
    "critical": 0.82,
}


@dataclass
class FunctionRisk:
    name: str
    lineno: int
    end_lineno: int
    raw_score: float            # bug heuristic score for this function in isolation
    complexity_weight: float    # share of total file cognitive complexity
    attributed_score: float     # raw_score * complexity_weight (localized signal)
    cognitive_complexity: int
    cyclomatic_complexity: int
    sloc: int
    rank: int
    risk_level: str             # "low" | "medium" | "high" | "critical"
    reason: str

    def to_dict(self) -> dict:
        return {
            "name":                 self.name,
            "lineno":               self.lineno,
            "end_lineno":           self.end_lineno,
            "raw_score":            round(self.raw_score, 4),
            "complexity_weight":    round(self.complexity_weight, 4),
            "attributed_score":     round(self.attributed_score, 4),
            "cognitive_complexity": self.cognitive_complexity,
            "cyclomatic_complexity":self.cyclomatic_complexity,
            "sloc":                 self.sloc,
            "rank":                 self.rank,
            "risk_level":           self.risk_level,
            "reason":               self.reason,
        }


@dataclass
class FunctionRiskResult:
    functions: list[FunctionRisk]
    top_k: list[FunctionRisk]
    file_risk_score: float
    total_functions: int
    n_high_risk: int
    localization_available: bool = True

    def to_dict(self) -> dict:
        return {
            "functions":             [f.to_dict() for f in self.functions],
            "top_k":                 [f.to_dict() for f in self.top_k],
            "file_risk_score":       round(self.file_risk_score, 4),
            "total_functions":       self.total_functions,
            "n_high_risk":           self.n_high_risk,
            "localization_available":self.localization_available,
        }


def _level(score: float) -> str:
    if score >= _HEURISTIC_THRESHOLDS["critical"]:
        return "critical"
    if score >= _HEURISTIC_THRESHOLDS["high"]:
        return "high"
    if score >= _HEURISTIC_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _reason(func_risk: float, cog: int, cc: int, sloc: int, weight: float) -> str:
    parts = []
    if cog > 15:
        parts.append(f"cognitive complexity {cog} (high)")
    elif cog > 8:
        parts.append(f"cognitive complexity {cog} (moderate)")
    if cc > 10:
        parts.append(f"cyclomatic complexity {cc}")
    if sloc > 50:
        parts.append(f"{sloc} lines")
    if weight > 0.3:
        parts.append(f"accounts for {weight*100:.0f}% of file complexity")
    if not parts:
        parts.append("elevated structural complexity")
    return "; ".join(parts)


def _heuristic_bug_score(cog: int, cc: int, sloc: int, n_params: int) -> float:
    """
    Heuristic bug probability for a single function.

    Based on the thresholds calibrated in bug_predictor.py for the
    no-git-metadata path.  Uses cognitive complexity as the primary
    signal (strongest predictor in the static-only ablation).
    """
    score = 0.10  # base rate

    # Cognitive complexity contribution (primary)
    if cog > 30:
        score += 0.55
    elif cog > 20:
        score += 0.40
    elif cog > 12:
        score += 0.25
    elif cog > 6:
        score += 0.12

    # Cyclomatic complexity contribution (secondary)
    if cc > 15:
        score += 0.15
    elif cc > 8:
        score += 0.08

    # Size contribution
    if sloc > 100:
        score += 0.08
    elif sloc > 50:
        score += 0.04

    # Parameter count (long parameter list smell)
    if n_params > 6:
        score += 0.06
    elif n_params > 4:
        score += 0.03

    return float(np.clip(score, 0.0, 1.0))


class FunctionRiskLocalizer:
    """
    Decomposes file-level bug risk to function granularity.

    No ML checkpoint needed — uses the heuristic scoring path
    from the bug predictor combined with per-function AST analysis.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def localize(self, source: str) -> FunctionRiskResult:
        """
        Parse source, extract per-function features, score each function,
        and return ranked attribution results.

        Args:
            source: Python source code string.

        Returns:
            FunctionRiskResult with ranked per-function risk scores.
        """
        try:
            return self._localize_internal(source)
        except SyntaxError:
            return FunctionRiskResult(
                functions=[], top_k=[], file_risk_score=0.0,
                total_functions=0, n_high_risk=0,
                localization_available=False,
            )
        except Exception as e:
            logger.warning("FunctionRiskLocalizer failed: %s", e)
            return FunctionRiskResult(
                functions=[], top_k=[], file_risk_score=0.0,
                total_functions=0, n_high_risk=0,
                localization_available=False,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _localize_internal(self, source: str) -> FunctionRiskResult:
        tree = ast.parse(source)
        src_lines = source.splitlines()

        function_data = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not hasattr(node, "end_lineno"):
                continue

            start = node.lineno - 1
            end   = node.end_lineno
            if end - start < 2:
                continue

            snippet = "\n".join(src_lines[start:end])
            metrics = self._compute_function_metrics(snippet, node)
            if metrics is None:
                continue

            cog, cc, sloc, n_params = metrics
            raw_score = _heuristic_bug_score(cog, cc, sloc, n_params)

            function_data.append({
                "name":    node.name,
                "lineno":  node.lineno,
                "end_lineno": end,
                "cog":     cog,
                "cc":      cc,
                "sloc":    sloc,
                "n_params":n_params,
                "raw_score": raw_score,
            })

        if not function_data:
            return FunctionRiskResult(
                functions=[], top_k=[], file_risk_score=0.0,
                total_functions=0, n_high_risk=0,
            )

        # Complexity-weighted attribution
        total_cog = sum(max(fd["cog"], 1) for fd in function_data)

        results: list[FunctionRisk] = []
        for fd in function_data:
            weight = max(fd["cog"], 1) / total_cog
            attributed = fd["raw_score"] * weight
            results.append(FunctionRisk(
                name=fd["name"],
                lineno=fd["lineno"],
                end_lineno=fd["end_lineno"],
                raw_score=round(fd["raw_score"], 4),
                complexity_weight=round(weight, 4),
                attributed_score=round(attributed, 4),
                cognitive_complexity=fd["cog"],
                cyclomatic_complexity=fd["cc"],
                sloc=fd["sloc"],
                rank=0,
                risk_level=_level(attributed),
                reason=_reason(fd["raw_score"], fd["cog"], fd["cc"], fd["sloc"], weight),
            ))

        results.sort(key=lambda r: r.attributed_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        # Aggregate file-level score = sum of attributed scores (bounded)
        file_score = float(np.clip(sum(r.attributed_score for r in results), 0.0, 1.0))
        n_high     = sum(1 for r in results if r.attributed_score > 0.30)

        return FunctionRiskResult(
            functions=results,
            top_k=results[:self.top_k],
            file_risk_score=round(file_score, 4),
            total_functions=len(results),
            n_high_risk=n_high,
        )

    def _compute_function_metrics(
        self, snippet: str, node: ast.FunctionDef
    ) -> Optional[tuple[int, int, int, int]]:
        """Return (cognitive_complexity, cyclomatic_complexity, sloc, n_params)."""
        try:
            from features.code_metrics import compute_all_metrics
            m = compute_all_metrics(snippet)
            cog   = int(getattr(m, "cognitive_complexity", 0) or 0)
            cc    = int(getattr(m, "cyclomatic_complexity", 0) or 0)
            sloc  = int(getattr(m.lines, "sloc", 0) or 0)
        except Exception:
            # Fallback: rough estimates from AST
            stmts = sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))
            branches = sum(
                1 for n in ast.walk(node)
                if isinstance(n, (ast.If, ast.For, ast.While, ast.Try,
                                  ast.ExceptHandler, ast.With, ast.Assert))
            )
            cog  = branches
            cc   = branches + 1
            sloc = stmts

        n_params = len(node.args.args)
        return cog, cc, sloc, n_params
