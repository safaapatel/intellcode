"""
Technical Debt Estimator
Aggregates all analysis results into a SQALE-inspired technical debt estimate.

The SQALE (Software Quality Assessment based on Lifecycle Expectations) model
assigns remediation time (in minutes) to each code quality issue found.

Debt Categories:
    security        — security vulnerabilities (critical/high/medium/low)
    reliability     — bug risk from bug predictor
    maintainability — complexity, long methods, deep nesting
    duplication     — code clone pairs
    dead_code       — unused/unreachable code
    style           — pattern violations, lines over 80 chars

Each category has a debt score (minutes) and a rating (A–E).
The overall debt rating is the worst of all categories.

Usage:
    estimator = TechnicalDebtEstimator()
    result = estimator.estimate(source, analysis_results)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Remediation time constants (minutes)
# ---------------------------------------------------------------------------

# Security
SEC_CRITICAL = 240   # 4 hours
SEC_HIGH     = 120   # 2 hours
SEC_MEDIUM   = 30
SEC_LOW      = 10

# Reliability (bug risk)
BUG_CRITICAL = 120
BUG_HIGH     = 60
BUG_MEDIUM   = 30

# Maintainability
CC_PER_UNIT_OVER_10  = 20   # per function with CC > 10
CC_PER_UNIT_OVER_20  = 45   # per function with CC > 20
LONG_METHOD          = 30   # per method > 40 lines
DEEP_NESTING         = 25   # per function with nesting > 3
MANY_PARAMS          = 15   # per function with > 4 params
COMPLEX_CONDITION    = 10   # per complex boolean condition

# Duplication
CLONE_TYPE1 = 30    # exact clone pair — consolidate
CLONE_TYPE2 = 20    # renamed clone pair
CLONE_TYPE3 = 15    # near-miss clone pair

# Dead code
DEAD_UNREACHABLE  = 5
DEAD_UNUSED_FUNC  = 10
DEAD_UNUSED_VAR   = 5
DEAD_UNUSED_IMPORT= 2
DEAD_EMPTY_EXCEPT = 15

# Style
STYLE_PATTERN_VIOLATION = 5
STYLE_LINE_OVER_80 = 1   # per line


# ---------------------------------------------------------------------------
# Rating thresholds (minutes of debt)
# ---------------------------------------------------------------------------

def _rating(minutes: int) -> str:
    """Convert debt minutes to SQALE letter rating."""
    if minutes <= 5:
        return "A"
    elif minutes <= 30:
        return "B"
    elif minutes <= 60:
        return "C"
    elif minutes <= 120:
        return "D"
    else:
        return "E"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DebtCategory:
    name: str
    debt_minutes: int
    rating: str
    items: list[str]    # human-readable breakdown lines

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TechnicalDebtResult:
    total_debt_minutes: int
    overall_rating: str         # A–E (worst of all categories)
    categories: list[DebtCategory]
    interest_per_day: float     # debt growth estimate (minutes/day)
    payoff_days: float          # how long to fix everything at 4h/day
    summary: str

    def to_dict(self) -> dict:
        return {
            "total_debt_minutes": self.total_debt_minutes,
            "overall_rating": self.overall_rating,
            "categories": [c.to_dict() for c in self.categories],
            "interest_per_day": round(self.interest_per_day, 1),
            "payoff_days": round(self.payoff_days, 2),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class TechnicalDebtEstimator:
    """
    Aggregates existing ML analysis results into a technical debt estimate.

    Can work standalone (from source code only) or with pre-computed results
    from other models to avoid re-running analysis.

    Args:
        source: Python source code
        security_result: dict from /analyze/security endpoint (optional)
        complexity_result: dict from /analyze/complexity endpoint (optional)
        bug_result: dict from /analyze/bugs endpoint (optional)
        clone_result: CloneDetectionResult or dict (optional)
        dead_code_result: DeadCodeResult or dict (optional)
        refactoring_result: RefactoringResult or dict (optional)
    """

    def estimate(
        self,
        source: str,
        security_result: Optional[dict] = None,
        complexity_result: Optional[dict] = None,
        bug_result: Optional[dict] = None,
        clone_result=None,
        dead_code_result=None,
        refactoring_result=None,
    ) -> TechnicalDebtResult:

        categories: list[DebtCategory] = []

        # --- Security ---
        sec = self._estimate_security(security_result)
        categories.append(sec)

        # --- Reliability (Bug Risk) ---
        rel = self._estimate_reliability(bug_result)
        categories.append(rel)

        # --- Maintainability ---
        maint = self._estimate_maintainability(source, complexity_result, refactoring_result)
        categories.append(maint)

        # --- Duplication ---
        dup = self._estimate_duplication(clone_result)
        categories.append(dup)

        # --- Dead Code ---
        dead = self._estimate_dead_code(dead_code_result)
        categories.append(dead)

        # --- Style ---
        style = self._estimate_style(source, complexity_result)
        categories.append(style)

        total = sum(c.debt_minutes for c in categories)
        rating_order = ["E", "D", "C", "B", "A"]
        worst_rating = min(
            (c.rating for c in categories),
            key=lambda r: rating_order.index(r) if r in rating_order else 0,
        )

        # Interest: severity-weighted debt growth (higher-rated categories compound faster).
        # E=3x, D=2x, C=1x, B=0.5x, A=0x — 0.5% base rate per day.
        # This means a critical security violation accrues ~6x more interest than a style issue.
        _severity_weight = {"E": 3.0, "D": 2.0, "C": 1.0, "B": 0.5, "A": 0.0}
        weighted_debt = sum(
            c.debt_minutes * _severity_weight.get(c.rating, 1.0)
            for c in categories
        )
        interest_per_day = round(weighted_debt * 0.005, 1)
        # Payoff: assume 4 hours/day of remediation work
        payoff_days = total / 240 if total > 0 else 0.0

        summary = (
            f"Technical Debt: {total} min "
            f"({total // 60}h {total % 60}min) — Rating {worst_rating}. "
            f"Estimated payoff: {payoff_days:.1f} working day(s)."
        )

        return TechnicalDebtResult(
            total_debt_minutes=total,
            overall_rating=worst_rating,
            categories=categories,
            interest_per_day=interest_per_day,
            payoff_days=payoff_days,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Category estimators
    # ------------------------------------------------------------------

    def _estimate_security(self, result: Optional[dict]) -> DebtCategory:
        minutes = 0
        items = []

        if result:
            summary = result.get("summary", {})
            n_crit = summary.get("critical", 0)
            n_high = summary.get("high", 0)
            n_med = summary.get("medium", 0)
            n_low = summary.get("low", 0)
            if n_crit:
                m = n_crit * SEC_CRITICAL
                minutes += m
                items.append(f"{n_crit} critical vulnerability(s) × {SEC_CRITICAL} min = {m} min")
            if n_high:
                m = n_high * SEC_HIGH
                minutes += m
                items.append(f"{n_high} high vulnerability(s) × {SEC_HIGH} min = {m} min")
            if n_med:
                m = n_med * SEC_MEDIUM
                minutes += m
                items.append(f"{n_med} medium vulnerability(s) × {SEC_MEDIUM} min = {m} min")
            if n_low:
                m = n_low * SEC_LOW
                minutes += m
                items.append(f"{n_low} low vulnerability(s) × {SEC_LOW} min = {m} min")
        else:
            items.append("No security analysis result provided")

        return DebtCategory(
            name="security",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items or ["No security issues found"],
        )

    def _estimate_reliability(self, result: Optional[dict]) -> DebtCategory:
        minutes = 0
        items = []

        if result:
            risk = result.get("risk_level", "low")
            prob = result.get("bug_probability", 0.0)
            if risk == "critical":
                minutes = BUG_CRITICAL
                items.append(f"Critical bug risk (p={prob:.2f}) → {BUG_CRITICAL} min")
            elif risk == "high":
                minutes = BUG_HIGH
                items.append(f"High bug risk (p={prob:.2f}) → {BUG_HIGH} min")
            elif risk == "medium":
                minutes = BUG_MEDIUM
                items.append(f"Medium bug risk (p={prob:.2f}) → {BUG_MEDIUM} min")
            else:
                items.append(f"Low bug risk (p={prob:.2f}) → 0 min")
        else:
            items.append("No bug prediction result provided")

        return DebtCategory(
            name="reliability",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items,
        )

    def _estimate_maintainability(
        self,
        source: str,
        complexity_result: Optional[dict],
        refactoring_result=None,
    ) -> DebtCategory:
        minutes = 0
        items = []

        if complexity_result:
            n_complex = complexity_result.get("n_complex_functions", 0)
            n_long = complexity_result.get("n_long_functions", 0)
            cc = complexity_result.get("cyclomatic", 0)

            if n_complex:
                m = n_complex * CC_PER_UNIT_OVER_10
                minutes += m
                items.append(f"{n_complex} complex function(s) × {CC_PER_UNIT_OVER_10} min = {m} min")
            if n_long:
                m = n_long * LONG_METHOD
                minutes += m
                items.append(f"{n_long} long function(s) × {LONG_METHOD} min = {m} min")
            if cc > 30:
                m = CC_PER_UNIT_OVER_20
                minutes += m
                items.append(f"Very high cyclomatic complexity ({cc}) → +{m} min")

        if refactoring_result:
            # Use pre-computed refactoring effort
            r = refactoring_result if isinstance(refactoring_result, dict) else refactoring_result.to_dict()
            effort = r.get("total_effort_minutes", 0)
            # Cap at 4h to avoid double-counting with complexity
            capped = min(effort, 240)
            if capped:
                minutes += capped
                items.append(f"Refactoring effort: {capped} min")
        elif not complexity_result:
            # Fallback: basic AST analysis
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        n_lines = getattr(node, "end_lineno", node.lineno) - node.lineno
                        if n_lines > 40:
                            minutes += LONG_METHOD
                            items.append(f"Long method '{node.name}' → {LONG_METHOD} min")
            except SyntaxError:
                pass

        if not items:
            items.append("No maintainability issues found")

        return DebtCategory(
            name="maintainability",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items,
        )

    def _estimate_duplication(self, result) -> DebtCategory:
        minutes = 0
        items = []

        if result is not None:
            r = result if isinstance(result, dict) else result.to_dict()
            clones = r.get("clones", [])
            for clone in clones:
                ct = clone.get("clone_type", "type3")
                if ct == "type1":
                    minutes += CLONE_TYPE1
                    items.append(
                        f"Exact clone: '{clone.get('block_a')}' ↔ '{clone.get('block_b')}' "
                        f"→ {CLONE_TYPE1} min"
                    )
                elif ct == "type2":
                    minutes += CLONE_TYPE2
                    items.append(
                        f"Renamed clone: '{clone.get('block_a')}' ↔ '{clone.get('block_b')}' "
                        f"→ {CLONE_TYPE2} min"
                    )
                else:
                    minutes += CLONE_TYPE3
                    items.append(
                        f"Near-miss clone ({clone.get('similarity', 0):.0%}): "
                        f"'{clone.get('block_a')}' ↔ '{clone.get('block_b')}' "
                        f"→ {CLONE_TYPE3} min"
                    )
        else:
            items.append("No clone detection result provided")

        if not items or (len(items) == 1 and "provided" not in items[0] and minutes == 0):
            items = ["No code clones detected"]

        return DebtCategory(
            name="duplication",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items,
        )

    def _estimate_dead_code(self, result) -> DebtCategory:
        minutes = 0
        items = []

        if result is not None:
            r = result if isinstance(result, dict) else result.to_dict()
            for issue in r.get("issues", []):
                itype = issue.get("issue_type", "")
                if itype == "unreachable_code":
                    minutes += DEAD_UNREACHABLE
                    items.append(f"Unreachable code ({issue.get('location')}) → {DEAD_UNREACHABLE} min")
                elif itype == "unused_function":
                    minutes += DEAD_UNUSED_FUNC
                    items.append(f"Unused function '{issue.get('title', '')}' → {DEAD_UNUSED_FUNC} min")
                elif itype == "unused_variable":
                    minutes += DEAD_UNUSED_VAR
                    items.append(f"Unused variable ({issue.get('location')}) → {DEAD_UNUSED_VAR} min")
                elif itype == "unused_import":
                    minutes += DEAD_UNUSED_IMPORT
                    items.append(f"Unused import ({issue.get('location')}) → {DEAD_UNUSED_IMPORT} min")
                elif itype == "empty_except":
                    minutes += DEAD_EMPTY_EXCEPT
                    items.append(f"Empty except block ({issue.get('location')}) → {DEAD_EMPTY_EXCEPT} min")
        else:
            items.append("No dead code analysis result provided")

        if not items or (len(items) == 1 and "provided" not in items[0] and minutes == 0):
            items = ["No dead code issues found"]

        return DebtCategory(
            name="dead_code",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items,
        )

    def _estimate_style(
        self,
        source: str,
        complexity_result: Optional[dict],
    ) -> DebtCategory:
        minutes = 0
        items = []

        if complexity_result:
            n_over_80 = complexity_result.get("n_lines_over_80", 0)
            if n_over_80:
                m = n_over_80 * STYLE_LINE_OVER_80
                minutes += m
                items.append(f"{n_over_80} line(s) over 80 chars × {STYLE_LINE_OVER_80} min = {m} min")
        else:
            # Fallback: count manually
            n_over_80 = sum(1 for line in source.splitlines() if len(line) > 80)
            if n_over_80:
                m = n_over_80 * STYLE_LINE_OVER_80
                minutes += m
                items.append(f"{n_over_80} line(s) over 80 chars → {m} min")

        if not items:
            items.append("No style issues found")

        return DebtCategory(
            name="style",
            debt_minutes=minutes,
            rating=_rating(minutes),
            items=items,
        )
