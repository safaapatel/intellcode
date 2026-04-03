"""
IntelliCode Review — FastAPI Backend
Serves all ML model predictions via a REST API.

Run:
    cd backend
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /models
    POST /analyze
    POST /analyze/security
    POST /analyze/complexity
    POST /analyze/patterns
    POST /analyze/bugs
    POST /analyze/clones
    POST /analyze/refactoring
    POST /analyze/dead-code
    POST /analyze/debt
    POST /analyze/docs
    POST /analyze/performance
    POST /analyze/dependencies
    POST /analyze/readability
"""

from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import hmac
import json as _json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from features.code_metrics import compute_all_metrics, compute_metrics_for_language
from features.ast_extractor import extract_ast_features
from features.security_patterns import scan_security_patterns, scan_security_patterns_for_language
from features.multi_lang.adapter import detect_language
from models.security_detection import EnsembleSecurityModel
from models.complexity_prediction import ComplexityPredictionModel, ComplexityResult
from models.bug_predictor import BugPredictionModel, GitMetadata
from models.pattern_recognition import PatternRFModel
from models.code_clone_detection import CodeCloneDetector
from models.refactoring_suggester import RefactoringSuggester
from models.dead_code_detector import DeadCodeDetector
from models.technical_debt import TechnicalDebtEstimator
from models.doc_quality_analyzer import DocQualityAnalyzer
from models.performance_analyzer import PerformanceAnalyzer
from models.dependency_analyzer import DependencyAnalyzer
from models.readability_scorer import ReadabilityScorer
from models.multi_task_model import MultiTaskCodeModel, MultiTaskPrediction


# ---------------------------------------------------------------------------
# Model singleton registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    security: EnsembleSecurityModel | None = None
    complexity: ComplexityPredictionModel | None = None
    bug_predictor: BugPredictionModel | None = None
    pattern_model: PatternRFModel | None = None
    clone_detector: CodeCloneDetector | None = None
    refactoring_suggester: RefactoringSuggester | None = None
    dead_code_detector: DeadCodeDetector | None = None
    debt_estimator: TechnicalDebtEstimator | None = None
    doc_analyzer: DocQualityAnalyzer | None = None
    performance_analyzer: PerformanceAnalyzer | None = None
    dependency_analyzer: DependencyAnalyzer | None = None
    readability_scorer: ReadabilityScorer | None = None
    multi_task: MultiTaskCodeModel | None = None
    load_errors: dict[str, str] = {}


registry = ModelRegistry()
logger = logging.getLogger(__name__)

# Thread-pool for parallel CPU-bound model inference (one worker per logical CPU,
# capped at 8 to avoid excessive context-switching on smaller machines)
_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))

# ---------------------------------------------------------------------------
# Simple in-memory response cache (hash of code+filename → response dict)
# ---------------------------------------------------------------------------
_analysis_cache: dict[str, dict] = {}
_specialist_cache: dict[str, dict] = {}

# Simple in-memory rate limiter: IP → deque of request timestamps
_rate_limit_store: dict[str, collections.deque] = {}
_RATE_LIMIT_REQUESTS = 20
_RATE_LIMIT_WINDOW_S = 60


def _check_rate_limit(request: Request) -> None:
    """Raise HTTP 429 if the client exceeds _RATE_LIMIT_REQUESTS per minute."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _rate_limit_store.setdefault(client_ip, collections.deque())
    # Evict timestamps outside the sliding window
    while window and now - window[0] > _RATE_LIMIT_WINDOW_S:
        window.popleft()
    if len(window) >= _RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 20 requests/minute")
    window.append(now)


def _cache_key(code: str, filename: str, git_metadata: Optional[dict] = None) -> str:
    payload = f"{filename}\x00{code}\x00{str(sorted((git_metadata or {}).items()))}"
    return hashlib.md5(payload.encode()).hexdigest()


def _specialist_key(endpoint: str, code: str) -> str:
    return hashlib.md5(f"{endpoint}\x00{code}".encode()).hexdigest()


def _cache_put(store: dict, key: str, value: dict, max_size: int = 200) -> None:
    if len(store) >= max_size:
        store.pop(next(iter(store)))
    store[key] = value


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Loading ML models...")

    try:
        registry.security = EnsembleSecurityModel(
            checkpoint_dir="checkpoints/security"
        )
        logger.info("  [OK] Security detection model")
    except Exception as e:
        registry.load_errors["security"] = str(e)
        logger.warning("  [WARN] Security model: %s", e)

    try:
        registry.complexity = ComplexityPredictionModel(
            checkpoint_path="checkpoints/complexity/model.pkl"
        )
        logger.info("  [OK] Complexity prediction model")
    except Exception as e:
        registry.load_errors["complexity"] = str(e)
        logger.warning("  [WARN] Complexity model: %s", e)

    try:
        registry.bug_predictor = BugPredictionModel(
            checkpoint_dir="checkpoints/bug_predictor"
        )
        logger.info("  [OK] Bug prediction model")
    except Exception as e:
        registry.load_errors["bug_predictor"] = str(e)
        logger.warning("  [WARN] Bug predictor: %s", e)

    try:
        registry.pattern_model = PatternRFModel(
            checkpoint_path="checkpoints/pattern/rf_model.pkl"
        )
        if registry.pattern_model.ready:
            logger.info("  [OK] Pattern recognition model (RF)")
        else:
            logger.warning("  [WARN] Pattern model checkpoint not found — will return null")
    except Exception as e:
        registry.load_errors["pattern"] = str(e)
        logger.warning("  [WARN] Pattern model: %s", e)

    # Lightweight models — always load
    registry.clone_detector = CodeCloneDetector()
    logger.info("  [OK] Code clone detector")

    registry.refactoring_suggester = RefactoringSuggester()
    logger.info("  [OK] Refactoring suggester")

    registry.dead_code_detector = DeadCodeDetector()
    logger.info("  [OK] Dead code detector")

    registry.debt_estimator = TechnicalDebtEstimator()
    logger.info("  [OK] Technical debt estimator")

    registry.doc_analyzer = DocQualityAnalyzer()
    logger.info("  [OK] Documentation quality analyzer")

    registry.performance_analyzer = PerformanceAnalyzer()
    logger.info("  [OK] Performance hotspot analyzer")

    registry.dependency_analyzer = DependencyAnalyzer()
    logger.info("  [OK] Dependency & coupling analyzer")

    registry.readability_scorer = ReadabilityScorer()
    logger.info("  [OK] Code readability scorer")

    try:
        registry.multi_task = MultiTaskCodeModel()
        if registry.multi_task.ready:
            logger.info("  [OK] Multi-task model (mode=%s)", registry.multi_task.mode)
        else:
            logger.info("  [OK] Multi-task model (no checkpoint — fallback mode)")
    except Exception as e:
        logger.warning("  [WARN] Multi-task model failed to load: %s", e)
        registry.multi_task = None

    # Restore persisted feedback so stats survive restarts
    _feedback_store.extend(_load_feedback_from_disk())

    logger.info("All models ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IntelliCode Review API",
    description="AI-powered code analysis REST API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://safaapatel.github.io",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze", min_length=1, max_length=500_000)
    filename: str = Field(default="snippet.py", description="Filename for context")
    language: str = Field(default="python", description="Programming language")
    git_metadata: Optional[dict] = Field(
        default=None,
        description=(
            "Optional git metadata: {code_churn, author_count, "
            "file_age_days, n_past_bugs, commit_freq}"
        ),
    )


class SecurityFindingOut(BaseModel):
    vuln_type: str
    severity: str
    title: str
    description: str
    lineno: int
    snippet: str
    confidence: float
    cwe: str
    source: str


class ComplexityOut(BaseModel):
    score: float
    grade: str
    cyclomatic: int
    cognitive: int
    halstead_bugs: float
    maintainability_index: float
    sloc: int
    n_long_functions: int
    n_complex_functions: int
    n_lines_over_80: int
    breakdown: dict
    function_issues: list


class BugPredictionOut(BaseModel):
    bug_probability: float
    risk_level: str
    risk_factors: list[str]
    confidence: float
    static_score: float
    git_score: Optional[float]


class PatternOut(BaseModel):
    label: str
    confidence: float
    all_scores: dict


class FullAnalysisResponse(BaseModel):
    filename: str
    language: str
    duration_seconds: float
    security: dict
    complexity: ComplexityOut
    bug_prediction: BugPredictionOut
    patterns: Optional[PatternOut]
    clones: dict
    refactoring: dict
    dead_code: dict
    technical_debt: dict
    overall_score: int
    status: str            # "clean" | "action_required" | "critical"
    summary: str
    # Specialist results — included in /analyze so the frontend needs only one call
    docs: Optional[dict] = None
    performance: Optional[dict] = None
    dependencies: Optional[dict] = None
    readability: Optional[dict] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Returns service status and model readiness."""
    return {
        "status": "ok",
        "models": {
            "security": "ready" if registry.security else "unavailable",
            "complexity": "ready" if registry.complexity else "unavailable",
            "bug_predictor": "ready" if registry.bug_predictor else "unavailable",
            "pattern_recognition": "ready" if (registry.pattern_model and registry.pattern_model.ready) else "unavailable",
            "clone_detector": "ready" if registry.clone_detector else "unavailable",
            "refactoring_suggester": "ready" if registry.refactoring_suggester else "unavailable",
            "dead_code_detector": "ready" if registry.dead_code_detector else "unavailable",
            "debt_estimator": "ready" if registry.debt_estimator else "unavailable",
            "doc_analyzer": "ready" if registry.doc_analyzer else "unavailable",
            "performance_analyzer": "ready" if registry.performance_analyzer else "unavailable",
            "dependency_analyzer": "ready" if registry.dependency_analyzer else "unavailable",
            "readability_scorer": "ready" if registry.readability_scorer else "unavailable",
        },
        "errors": registry.load_errors,
        "cache": {
            "full_analysis_entries": len(_analysis_cache),
            "specialist_entries": len(_specialist_cache),
        },
    }


def _require_internal_key(request: Request):
    """Reject requests that don't supply the correct INTERNAL_API_KEY header.
    If INTERNAL_API_KEY is not configured, the endpoint is disabled (returns 503)
    to prevent unauthenticated cache-clearing in production.
    """
    secret = os.environ.get("INTERNAL_API_KEY", "")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="INTERNAL_API_KEY is not configured — cache clear is disabled. "
                   "Set the INTERNAL_API_KEY environment variable to enable this endpoint.",
        )
    if request.headers.get("X-Internal-Key", "") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/cache/clear", tags=["meta"])
def clear_cache(request: Request):
    """Clear the in-memory analysis cache (useful after model updates)."""
    _require_internal_key(request)
    _analysis_cache.clear()
    _specialist_cache.clear()
    return {"status": "ok", "message": "Cache cleared"}


@app.get("/stats", tags=["meta"])
def get_stats():
    """Aggregate statistics across all cached analyses."""
    models_ready = _count_ready()
    if not _analysis_cache:
        return {"total_analyses": 0, "avg_score": None, "models_ready": models_ready}
    scores = [v["overall_score"] for v in _analysis_cache.values() if v.get("overall_score") is not None]
    sec_findings = sum(
        len(v.get("security", {}).get("findings", []))
        for v in _analysis_cache.values()
    )
    return {
        "total_analyses": len(_analysis_cache),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "total_security_findings": sec_findings,
        "specialist_cache_entries": len(_specialist_cache),
        "models_ready": models_ready,
    }


@app.get("/models", tags=["meta"])
def list_models():
    """Returns metadata about each available ML model.
    Metrics are loaded from checkpoints/*/metrics.json when available,
    falling back to last-known values.
    """
    m_pattern  = _load_checkpoint_metrics("pattern")
    m_security = _load_checkpoint_metrics("security")
    m_complexity = _load_checkpoint_metrics("complexity")
    m_bug      = _load_checkpoint_metrics("bug_predictor")

    return {
        "models": [
            {
                "id": "pattern_recognition",
                "name": "Pattern Recognition Model",
                "architecture": "Random Forest (300 trees, CalibratedClassifierCV)",
                "status": "production",
                "accuracy": m_pattern.get("accuracy", 0.809),
                "auc": m_pattern.get("auc", 0.960),
                "cv_f1": m_pattern.get("cv_f1"),
                "tech_stack": ["scikit-learn", "Python AST"],
                "use_case": "Detects code smells, anti-patterns, style violations",
                "loaded": registry.pattern_model is not None and registry.pattern_model.ready,
            },
            {
                "id": "security_detection",
                "name": "Security Vulnerability Detection",
                "architecture": "Ensemble: Random Forest + 1D CNN",
                "status": "production",
                "auc": m_security.get("ensemble_auc", m_security.get("auc", 0.928)),
                "recall": m_security.get("recall", 0.54),
                "threshold": m_security.get("rf_threshold"),
                "tech_stack": ["scikit-learn", "PyTorch"],
                "use_case": "Detects SQL injection, XSS, hardcoded secrets, etc.",
                "loaded": registry.security is not None,
            },
            {
                "id": "complexity_prediction",
                "name": "Code Complexity Prediction",
                "architecture": "XGBoost Regressor",
                "status": "production",
                "r2_score": m_complexity.get("r2", m_complexity.get("test_r2", 1.0)),
                "rmse": m_complexity.get("rmse", m_complexity.get("test_rmse")),
                "tech_stack": ["XGBoost", "NumPy"],
                "use_case": "Predicts maintainability score (0–100)",
                "loaded": registry.complexity is not None,
            },
            {
                "id": "bug_predictor",
                "name": "Bug Prediction Model",
                "architecture": "LR + XGBoost ensemble",
                "status": "production",
                "auc": m_bug.get("ensemble_auc", m_bug.get("auc", 0.704)),
                "tech_stack": ["scikit-learn", "XGBoost"],
                "use_case": "Predicts bug likelihood from static + git features",
                "loaded": registry.bug_predictor is not None,
            },
            {
                "id": "code_clone_detection",
                "name": "Code Clone Detection",
                "architecture": "TF-IDF Embedder + Cosine Similarity (Type-1/2/3)",
                "status": "production",
                "tech_stack": ["Python AST", "NumPy"],
                "use_case": "Detects exact, renamed, and near-miss code duplicates",
                "loaded": registry.clone_detector is not None,
            },
            {
                "id": "refactoring_suggester",
                "name": "Refactoring Suggester",
                "architecture": "AST-based rule engine with effort scoring",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Suggests Extract Method, Reduce Nesting, Simplify Conditions, etc.",
                "loaded": registry.refactoring_suggester is not None,
            },
            {
                "id": "dead_code_detector",
                "name": "Dead Code Detector",
                "architecture": "AST visitor with definition/usage graph",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Detects unreachable code, unused imports, empty except blocks, etc.",
                "loaded": registry.dead_code_detector is not None,
            },
            {
                "id": "technical_debt",
                "name": "Technical Debt Estimator",
                "architecture": "SQALE-inspired aggregation model",
                "status": "production",
                "tech_stack": ["Python"],
                "use_case": "Estimates remediation time and A–E debt rating per category",
                "loaded": registry.debt_estimator is not None,
            },
            {
                "id": "doc_quality",
                "name": "Documentation Quality Analyzer",
                "architecture": "AST docstring parser with quality scoring",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Scores docstring coverage, completeness, and quality (A–F grade)",
                "loaded": registry.doc_analyzer is not None,
            },
            {
                "id": "performance_analyzer",
                "name": "Performance Hotspot Predictor",
                "architecture": "AST pattern matcher for anti-patterns",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Detects O(n²) loops, I/O in loops, string concat, mutable defaults, etc.",
                "loaded": registry.performance_analyzer is not None,
            },
            {
                "id": "dependency_analyzer",
                "name": "Dependency & Coupling Analyzer",
                "architecture": "Import graph analysis with coupling metrics",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Measures fan-out, instability, wildcard imports, coupling score",
                "loaded": registry.dependency_analyzer is not None,
            },
            {
                "id": "readability_scorer",
                "name": "Code Readability Scorer",
                "architecture": "Multi-dimension AST scoring model",
                "status": "production",
                "tech_stack": ["Python AST"],
                "use_case": "Scores naming, comments, structure, cognitive load → A–F grade",
                "loaded": registry.readability_scorer is not None,
            },
            {
                "id": "multi_task",
                "name": "Multi-Task Code Quality Model",
                "architecture": "Shared transformer encoder with 4 task heads (security, complexity, bugs, patterns)",
                "status": "experimental",
                "tech_stack": ["PyTorch", "Transformers"],
                "use_case": "Ablation study: unified MTL model vs 4 independent specialist models",
                "loaded": registry.multi_task is not None,
            },
        ],
        "multi_task": registry.multi_task is not None,
    }


def _load_checkpoint_metrics(task: str) -> dict:
    """Load metrics from checkpoints/<task>/metrics.json; return {} if missing."""
    metrics_path = Path(f"checkpoints/{task}/metrics.json")
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path) as f:
            return _json.load(f)
    except Exception as e:
        logger.warning("Could not load metrics for %s: %s", task, e)
        return {}


def _count_ready() -> int:
    """Count how many ML models are loaded and ready."""
    return sum(1 for m in [
        registry.security, registry.complexity, registry.bug_predictor,
        registry.pattern_model, registry.clone_detector, registry.refactoring_suggester,
        registry.dead_code_detector, registry.debt_estimator, registry.doc_analyzer,
        registry.performance_analyzer, registry.dependency_analyzer, registry.readability_scorer,
    ] if m is not None)


def _parse_git_meta(raw: Optional[dict]) -> Optional[GitMetadata]:
    """Parse optional git metadata dict into a GitMetadata dataclass."""
    if not raw:
        return None
    return GitMetadata(**{
        k: raw.get(k, 0)
        for k in ("code_churn", "author_count", "file_age_days", "n_past_bugs", "commit_freq")
    })


def _compute_overall_score(
    complexity_score: float,
    security_out: dict,
    clone_out: dict,
    bug_out,
    dead_out: dict,
    debt_result,
) -> int:
    """Shared weighted scoring formula used by both /analyze and /analyze/stream."""
    crit = security_out.get("summary", {}).get("critical", 0)
    high = security_out.get("summary", {}).get("high", 0)
    med  = security_out.get("summary", {}).get("medium", 0)
    security_score = max(0, 100 - crit * 20 - high * 10 - med * 4)
    clone_rate = clone_out.get("clone_rate", 0)
    clone_score = max(0, 100 - int(clone_rate * 120))
    dead_ratio  = dead_out.get("dead_ratio", 0)
    dead_score  = max(0, 100 - int(dead_ratio * 150))
    bug_prob    = bug_out.bug_probability if bug_out is not None else 0.5
    bug_score   = max(0, int((1 - bug_prob) * 100))
    debt_score  = {"A": 100, "B": 80, "C": 60, "D": 40, "E": 20}.get(debt_result.overall_rating, 50)
    return max(0, min(100, int(
        complexity_score * 0.25 +
        security_score   * 0.30 +
        clone_score      * 0.15 +
        bug_score        * 0.15 +
        dead_score       * 0.05 +
        debt_score       * 0.10
    )))


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["analysis"])
async def analyze_full(req: AnalyzeRequest, request: Request):
    """
    Full analysis: runs all ML models and returns a combined report.
    Independent models execute in parallel via ThreadPoolExecutor.
    Results are cached by (code + filename + git_metadata) hash.
    """
    _check_rate_limit(request)
    cache_key = _cache_key(req.code, req.filename, req.git_metadata)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    t_start = time.perf_counter()
    source = req.code
    language = detect_language(req.filename, req.language)
    # Pre-compute metrics with the language-aware adapter for non-Python languages.
    # Python uses the model's internal compute_all_metrics for full ML support.
    precomputed = compute_metrics_for_language(source, language) if language != "python" else None
    loop = asyncio.get_running_loop()

    def _run(fn, *args):
        return loop.run_in_executor(_EXECUTOR, fn, *args)

    # --- Wave 1: all independent models run in parallel ---
    try:
        (
            security_out,
            (complexity_out, complexity_result),
            (bug_out, bug_result),
            pattern_out,
            (clone_out, clone_result),
            (refactor_out, refactor_result),
            (dead_out, dead_result),
        ) = await asyncio.gather(
            _run(_step_security, source, language),
            _run(_step_complexity, source, precomputed),
            _run(functools.partial(_step_bug, source, req.git_metadata)),
            _run(_step_pattern, source),
            _run(_step_clones, source),
            _run(_step_refactoring, source),
            _run(_step_dead_code, source),
        )
    except RuntimeError as exc:
        # _step_complexity / _step_bug raise RuntimeError when model unavailable
        if "unavailable" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc))
        raise

    # --- Wave 2: debt (needs wave-1 results) + specialists (independent) in parallel ---
    (
        (debt_out, debt_result),
        docs_out,
        perf_out,
        deps_out,
        read_out,
    ) = await asyncio.gather(
        _run(functools.partial(
            _step_debt, source, security_out,
            complexity_result, bug_result, clone_result, dead_result, refactor_result,
        )),
        _run(_step_docs, source),
        _run(_step_performance, source),
        _run(_step_dependencies, source),
        _run(_step_readability, source),
    )

    # Cache specialist results too
    for key_sfx, out in [("docs", docs_out), ("performance", perf_out),
                          ("dependencies", deps_out), ("readability", read_out)]:
        sk = _specialist_key(key_sfx, source)
        _cache_put(_specialist_cache, sk, out)

    # --- Overall score (weighted across all available signals) ---
    overall_score = _compute_overall_score(
        complexity_out.score, security_out, clone_out, bug_out, dead_out, debt_result
    )

    # Status — critical only for actual security critical findings
    crit = security_out.get("summary", {}).get("critical", 0)
    high = security_out.get("summary", {}).get("high", 0)
    bug_prob = bug_out.bug_probability if bug_out is not None else 0.5
    if crit > 0:
        status = "critical"
    elif high > 0 or (complexity_out.score < 40 and debt_result.overall_rating in ("D", "E")):
        status = "action_required"
    elif complexity_out.score < 60 or debt_result.overall_rating in ("D", "E") or bug_prob > 0.7:
        status = "action_required"
    else:
        status = "clean"

    total_issues = security_out.get("summary", {}).get("total", 0)
    summary = (
        f"Found {total_issues} security issue(s). "
        f"Code quality: {complexity_out.score}/100 ({complexity_out.grade}). "
        f"Bug risk: {bug_out.risk_level}. "
        f"Clones: {len(clone_out.get('clones', []))}. "
        f"Debt: {debt_result.total_debt_minutes} min (Rating {debt_result.overall_rating})."
    )

    duration = round(time.perf_counter() - t_start, 3)

    response = FullAnalysisResponse(
        filename=req.filename,
        language=req.language,
        duration_seconds=duration,
        security=security_out,
        complexity=complexity_out,
        bug_prediction=bug_out,
        patterns=pattern_out,
        clones=clone_out,
        refactoring=refactor_out,
        dead_code=dead_out,
        technical_debt=debt_out,
        overall_score=overall_score,
        status=status,
        summary=summary,
        docs=docs_out or None,
        performance=perf_out or None,
        dependencies=deps_out or None,
        readability=read_out or None,
    )
    _cache_put(_analysis_cache, cache_key, response.model_dump())
    return response


@app.post("/analyze/security", tags=["analysis"])
def analyze_security(req: AnalyzeRequest):
    """Security vulnerability scan only."""
    t_start = time.perf_counter()

    language = detect_language(req.filename, req.language)
    if registry.security and language == "python":
        findings = registry.security.predict(req.code)
        vuln_score = registry.security.vulnerability_score(req.code)
        result = {
            "findings": [vars(f) for f in findings],
            "vulnerability_score": round(vuln_score, 3),
        }
    else:
        result = scan_security_patterns_for_language(req.code, language).to_dict()

    result["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return result


@app.post("/analyze/complexity", tags=["analysis"])
def analyze_complexity(req: AnalyzeRequest):
    """Complexity and maintainability analysis only."""
    t_start = time.perf_counter()

    language = detect_language(req.filename, req.language)
    if language != "python":
        metrics = compute_metrics_for_language(req.code, language)
        out, _ = _build_complexity_out_from_metrics(metrics)
        return {**out.model_dump(), "duration_seconds": round(time.perf_counter() - t_start, 3)}

    if not registry.complexity:
        raise HTTPException(status_code=503, detail="Complexity model unavailable")
    result = registry.complexity.predict(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/patterns", tags=["analysis"])
def analyze_patterns(req: AnalyzeRequest):
    """Pattern recognition (code smells / anti-patterns)."""
    t_start = time.perf_counter()

    if registry.pattern_model is None or not registry.pattern_model.ready:
        raise HTTPException(
            status_code=501,
            detail="Pattern model unavailable — run training/train_all.py first"
        )

    pred = registry.pattern_model.predict(req.code)
    return {
        "label": pred.label,
        "confidence": pred.confidence,
        "all_scores": pred.all_scores,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
    }


@app.post("/analyze/bugs", tags=["analysis"])
def analyze_bugs(req: AnalyzeRequest):
    """Bug probability prediction."""
    t_start = time.perf_counter()

    git_meta = _parse_git_meta(req.git_metadata)
    if registry.bug_predictor:
        result = registry.bug_predictor.predict(req.code, git_meta)
    else:
        raise HTTPException(status_code=503, detail="Bug prediction model unavailable")

    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/clones", tags=["analysis"])
def analyze_clones(req: AnalyzeRequest):
    """Code clone detection — finds Type-1, Type-2, and Type-3 duplicates."""
    t_start = time.perf_counter()
    result = registry.clone_detector.detect(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/refactoring", tags=["analysis"])
def analyze_refactoring(req: AnalyzeRequest):
    """Refactoring suggestions — actionable recommendations with effort estimates."""
    t_start = time.perf_counter()
    result = registry.refactoring_suggester.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/dead-code", tags=["analysis"])
def analyze_dead_code(req: AnalyzeRequest):
    """Dead code detection — unused imports, unreachable code, empty except blocks, etc."""
    t_start = time.perf_counter()
    result = registry.dead_code_detector.detect(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/debt", tags=["analysis"])
def analyze_debt(req: AnalyzeRequest):
    """
    Technical debt estimation.
    Returns the cached debt result from a prior /analyze call when available,
    otherwise runs just the lightweight analyses required by the debt estimator.
    """
    t_start = time.perf_counter()

    # Fast path: reuse cached full analysis result
    cache_key = _cache_key(req.code, req.filename, req.git_metadata)
    if cache_key in _analysis_cache:
        cached = _analysis_cache[cache_key]
        if "technical_debt" in cached:
            out = dict(cached["technical_debt"])
            out["duration_seconds"] = round(time.perf_counter() - t_start, 4)
            out["_cache_hit"] = True
            return out

    # Slow path: run only the models the debt estimator needs
    security_out = None
    if registry.security:
        findings = registry.security.predict(req.code)
        security_out = {
            "summary": {
                "critical": sum(1 for f in findings if f.severity == "critical"),
                "high": sum(1 for f in findings if f.severity == "high"),
                "medium": sum(1 for f in findings if f.severity == "medium"),
                "low": sum(1 for f in findings if f.severity == "low"),
            }
        }

    complexity_out = registry.complexity.predict(req.code).to_dict() if registry.complexity else None
    bug_out = registry.bug_predictor.predict(req.code).to_dict() if registry.bug_predictor else None
    clone_result = registry.clone_detector.detect(req.code)
    dead_result = registry.dead_code_detector.detect(req.code)
    refactor_result = registry.refactoring_suggester.analyze(req.code)

    result = registry.debt_estimator.estimate(
        source=req.code,
        security_result=security_out,
        complexity_result=complexity_out,
        bug_result=bug_out,
        clone_result=clone_result,
        dead_code_result=dead_result,
        refactoring_result=refactor_result,
    )

    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/docs", tags=["analysis"])
def analyze_docs(req: AnalyzeRequest):
    """Documentation quality — scores docstring coverage and completeness."""
    key = _specialist_key("docs", req.code)
    if key in _specialist_cache:
        return _specialist_cache[key]
    t_start = time.perf_counter()
    result = registry.doc_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/performance", tags=["analysis"])
def analyze_performance(req: AnalyzeRequest):
    """Performance hotspot detection — finds O(n²) loops, I/O in loops, etc."""
    key = _specialist_key("performance", req.code)
    if key in _specialist_cache:
        return _specialist_cache[key]
    t_start = time.perf_counter()
    result = registry.performance_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/dependencies", tags=["analysis"])
def analyze_dependencies(req: AnalyzeRequest):
    """Dependency & coupling analysis — fan-out, wildcard imports, coupling score."""
    key = _specialist_key("dependencies", req.code)
    if key in _specialist_cache:
        return _specialist_cache[key]
    t_start = time.perf_counter()
    result = registry.dependency_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/readability", tags=["analysis"])
def analyze_readability(req: AnalyzeRequest):
    """Code readability scoring — naming, comments, structure, cognitive load."""
    key = _specialist_key("readability", req.code)
    if key in _specialist_cache:
        return _specialist_cache[key]
    t_start = time.perf_counter()
    result = registry.readability_scorer.score(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


# ---------------------------------------------------------------------------
# Streaming analysis helpers (sync wrappers for asyncio.to_thread)
# ---------------------------------------------------------------------------

def _build_complexity_out_from_metrics(metrics) -> tuple:
    """Build ComplexityOut + ComplexityResult from a pre-computed CodeMetricsResult.

    Used for non-Python languages where the ML model's internal Python-AST
    metric extraction would fail. Scoring is purely rule-based.
    """
    score = 100.0
    cc = metrics.cyclomatic_complexity
    if cc > 10:
        score -= min(30, (cc - 10) * 1.5)
    cog = metrics.cognitive_complexity
    if cog > 15:
        score -= min(20, (cog - 15) * 0.8)
    if metrics.n_long_functions > 0:
        score -= min(15, metrics.n_long_functions * 3)
    if metrics.n_complex_functions > 0:
        score -= min(20, metrics.n_complex_functions * 4)
    lines_over = metrics.n_lines_over_80
    if lines_over > 20:
        score -= min(10, (lines_over - 20) * 0.2)
    score = max(0.0, min(100.0, score))

    def _grade(s):
        if s >= 85: return "A"
        if s >= 70: return "B"
        if s >= 55: return "C"
        if s >= 40: return "D"
        return "F"

    cr = ComplexityResult(
        score=round(score, 1),
        grade=_grade(score),
        cyclomatic=metrics.cyclomatic_complexity,
        cognitive=metrics.cognitive_complexity,
        halstead_bugs=round(metrics.halstead.bugs_delivered, 3),
        maintainability_index=round(metrics.maintainability_index, 1),
        sloc=metrics.lines.sloc,
        n_long_functions=metrics.n_long_functions,
        n_complex_functions=metrics.n_complex_functions,
        n_lines_over_80=metrics.n_lines_over_80,
        function_issues=[],
        breakdown={
            "cyclomatic_complexity": metrics.cyclomatic_complexity,
            "cognitive_complexity": metrics.cognitive_complexity,
            "halstead_volume": round(metrics.halstead.volume, 1),
            "halstead_difficulty": round(metrics.halstead.difficulty, 2),
            "maintainability_index": round(metrics.maintainability_index, 1),
            "sloc": metrics.lines.sloc,
            "lines_over_80": metrics.n_lines_over_80,
            "long_functions": metrics.n_long_functions,
            "complex_functions": metrics.n_complex_functions,
            "function_level": [],
        },
    )
    return ComplexityOut(**cr.to_dict()), cr


def _step_security(source: str, language: str = "python") -> dict:
    if registry.security and language == "python":
        # ML-based security detection is trained on Python only
        findings = registry.security.predict(source)
        sec_score = registry.security.vulnerability_score(source)
        return {
            "findings": [vars(f) for f in findings],
            "vulnerability_score": round(sec_score, 3),
            "summary": {
                "total": len(findings),
                "critical": sum(1 for f in findings if f.severity == "critical"),
                "high": sum(1 for f in findings if f.severity == "high"),
                "medium": sum(1 for f in findings if f.severity == "medium"),
                "low": sum(1 for f in findings if f.severity == "low"),
            },
        }
    return scan_security_patterns_for_language(source, language).to_dict()


def _step_complexity(source: str, precomputed=None):
    if precomputed is not None:
        # Non-Python: skip the Python-AST-based ML model, use rule-based scoring
        return _build_complexity_out_from_metrics(precomputed)
    if not registry.complexity:
        raise RuntimeError("Complexity model unavailable")
    r = registry.complexity.predict(source)
    return ComplexityOut(**r.to_dict()), r


def _step_bug(source: str, git_metadata: Optional[dict]):
    git_meta = _parse_git_meta(git_metadata)
    if not registry.bug_predictor:
        raise RuntimeError("Bug prediction model unavailable")
    r = registry.bug_predictor.predict(source, git_meta)
    return BugPredictionOut(**r.to_dict()), r


def _step_pattern(source: str):
    if registry.pattern_model and registry.pattern_model.ready:
        try:
            pred = registry.pattern_model.predict(source)
            return PatternOut(label=pred.label, confidence=pred.confidence, all_scores=pred.all_scores)
        except Exception as e:
            logger.warning("Pattern model prediction failed: %s", e)
    return None


def _step_clones(source: str):
    r = registry.clone_detector.detect(source)
    return r.to_dict(), r


def _step_refactoring(source: str):
    r = registry.refactoring_suggester.analyze(source)
    return r.to_dict(), r


def _step_dead_code(source: str):
    r = registry.dead_code_detector.detect(source)
    return r.to_dict(), r


def _step_debt(source, sec, cpx_r, bug_r, clone_r, dead_r, refactor_r):
    r = registry.debt_estimator.estimate(
        source=source,
        security_result=sec,
        complexity_result=cpx_r.to_dict(),
        bug_result=bug_r.to_dict(),
        clone_result=clone_r,
        dead_code_result=dead_r,
        refactoring_result=refactor_r,
    )
    return r.to_dict(), r


def _step_docs(source: str):
    r = registry.doc_analyzer.analyze(source)
    return r.to_dict() if r else {}


def _step_performance(source: str):
    r = registry.performance_analyzer.analyze(source)
    return r.to_dict() if r else {}


def _step_dependencies(source: str):
    r = registry.dependency_analyzer.analyze(source)
    return r.to_dict() if r else {}


def _step_readability(source: str):
    r = registry.readability_scorer.score(source)
    return r.to_dict() if r else {}


# ---------------------------------------------------------------------------
# SSE streaming analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze/stream", tags=["analysis"])
async def analyze_stream(req: AnalyzeRequest):
    """
    Streaming analysis: yields Server-Sent Events as each model completes.
    Final event has status='complete' and contains the full merged result.
    """

    async def generate():
        def evt(data: dict) -> str:
            return f"data: {_json.dumps(data)}\n\n"

        # --- Cache hit: replay all steps instantly then emit result ---
        cache_key = _cache_key(req.code, req.filename, req.git_metadata)
        if cache_key in _analysis_cache:
            cached = _analysis_cache[cache_key]
            steps = [
                "Security Detection", "Complexity Analysis", "Bug Prediction",
                "Pattern Recognition", "Clone Detection", "Refactoring Analysis",
                "Dead Code Detection", "Technical Debt", "Documentation Quality",
                "Performance Analysis", "Dependency Analysis", "Readability Scoring",
            ]
            for i, name in enumerate(steps):
                yield evt({"step": name, "progress": int((i + 1) / len(steps) * 100), "status": "done"})
            yield evt({"status": "complete", "result": cached})
            return

        t_start = time.perf_counter()
        source = req.code
        language = detect_language(req.filename, req.language)
        precomputed = compute_metrics_for_language(source, language) if language != "python" else None

        # 1 — Security
        yield evt({"step": "Security Detection", "progress": 8, "status": "running"})
        security_out = await asyncio.to_thread(_step_security, source, language)
        yield evt({"step": "Security Detection", "progress": 8, "status": "done"})

        # 2 — Complexity
        yield evt({"step": "Complexity Analysis", "progress": 17, "status": "running"})
        complexity_out, complexity_result = await asyncio.to_thread(_step_complexity, source, precomputed)
        yield evt({"step": "Complexity Analysis", "progress": 17, "status": "done"})

        # 3 — Bug prediction
        yield evt({"step": "Bug Prediction", "progress": 25, "status": "running"})
        bug_out, bug_result = await asyncio.to_thread(_step_bug, source, req.git_metadata)
        yield evt({"step": "Bug Prediction", "progress": 25, "status": "done"})

        # 4 — Pattern
        yield evt({"step": "Pattern Recognition", "progress": 33, "status": "running"})
        pattern_out = await asyncio.to_thread(_step_pattern, source)
        yield evt({"step": "Pattern Recognition", "progress": 33, "status": "done"})

        # 5 — Clones
        yield evt({"step": "Clone Detection", "progress": 42, "status": "running"})
        clone_out, clone_result = await asyncio.to_thread(_step_clones, source)
        yield evt({"step": "Clone Detection", "progress": 42, "status": "done"})

        # 6 — Refactoring
        yield evt({"step": "Refactoring Analysis", "progress": 50, "status": "running"})
        refactor_out, refactor_result = await asyncio.to_thread(_step_refactoring, source)
        yield evt({"step": "Refactoring Analysis", "progress": 50, "status": "done"})

        # 7 — Dead code
        yield evt({"step": "Dead Code Detection", "progress": 58, "status": "running"})
        dead_out, dead_result = await asyncio.to_thread(_step_dead_code, source)
        yield evt({"step": "Dead Code Detection", "progress": 58, "status": "done"})

        # 8 — Technical debt (depends on prev results)
        yield evt({"step": "Technical Debt", "progress": 67, "status": "running"})
        debt_out, debt_result = await asyncio.to_thread(
            _step_debt, source, security_out, complexity_result,
            bug_result, clone_result, dead_result, refactor_result
        )
        yield evt({"step": "Technical Debt", "progress": 67, "status": "done"})

        # 9 — Docs
        yield evt({"step": "Documentation Quality", "progress": 75, "status": "running"})
        docs_out = await asyncio.to_thread(_step_docs, source)
        yield evt({"step": "Documentation Quality", "progress": 75, "status": "done"})

        # 10 — Performance
        yield evt({"step": "Performance Analysis", "progress": 83, "status": "running"})
        perf_out = await asyncio.to_thread(_step_performance, source)
        yield evt({"step": "Performance Analysis", "progress": 83, "status": "done"})

        # 11 — Dependencies
        yield evt({"step": "Dependency Analysis", "progress": 92, "status": "running"})
        deps_out = await asyncio.to_thread(_step_dependencies, source)
        yield evt({"step": "Dependency Analysis", "progress": 92, "status": "done"})

        # 12 — Readability
        yield evt({"step": "Readability Scoring", "progress": 100, "status": "running"})
        readability_out = await asyncio.to_thread(_step_readability, source)
        yield evt({"step": "Readability Scoring", "progress": 100, "status": "done"})

        # --- Overall score (same formula as /analyze) ---
        overall_score = _compute_overall_score(
            complexity_out.score, security_out, clone_out, bug_out, dead_out, debt_result
        )
        crit = security_out.get("summary", {}).get("critical", 0)
        high = security_out.get("summary", {}).get("high", 0)
        bug_prob = bug_out.bug_probability if bug_out is not None else 0.5

        if crit > 0:
            status = "critical"
        elif high > 0 or (complexity_out.score < 40 and debt_result.overall_rating in ("D", "E")):
            status = "action_required"
        elif complexity_out.score < 60 or debt_result.overall_rating in ("D", "E") or bug_prob > 0.7:
            status = "action_required"
        else:
            status = "clean"

        total_issues = security_out.get("summary", {}).get("total", 0)
        summary = (
            f"Found {total_issues} security issue(s). "
            f"Code quality: {complexity_out.score}/100 ({complexity_out.grade}). "
            f"Bug risk: {bug_out.risk_level}. "
            f"Clones: {len(clone_out.get('clones', []))}. "
            f"Debt: {debt_result.total_debt_minutes} min (Rating {debt_result.overall_rating})."
        )

        response_dict = FullAnalysisResponse(
            filename=req.filename,
            language=req.language,
            duration_seconds=round(time.perf_counter() - t_start, 3),
            security=security_out,
            complexity=complexity_out,
            bug_prediction=bug_out,
            patterns=pattern_out,
            clones=clone_out,
            refactoring=refactor_out,
            dead_code=dead_out,
            technical_debt=debt_out,
            overall_score=overall_score,
            status=status,
            summary=summary,
        ).model_dump()

        # Merge in specialist results so frontend gets the full picture
        response_dict["docs"] = docs_out
        response_dict["performance"] = perf_out
        response_dict["dependencies"] = deps_out
        response_dict["readability"] = readability_out

        _cache_put(_analysis_cache, cache_key, response_dict)
        yield evt({"status": "complete", "result": response_dict})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Quick analysis endpoint (fast models only, no ML training required)
# ---------------------------------------------------------------------------

@app.post("/analyze/quick", tags=["analysis"])
def analyze_quick(req: AnalyzeRequest):
    """
    Quick analysis using only fast heuristic/rules-based checks.
    Runs: security pattern scan, complexity metrics, readability scoring.
    Returns results in < 500 ms (no heavyweight ML inference).
    """
    source = req.code
    language = detect_language(req.filename, req.language)
    result: dict = {"filename": req.filename, "language": language, "mode": "quick"}

    # Security pattern scan (regex-based, very fast)
    try:
        sec_scan = scan_security_patterns_for_language(source, language)
        result["security_patterns"] = {
            "findings": [vars(f) for f in sec_scan.findings],
            "count": len(sec_scan.findings),
            "has_critical": bool(sec_scan.critical),
        }
    except Exception as e:
        result["security_patterns"] = {"error": str(e)}

    # Complexity metrics (pure computation, no ML)
    try:
        metrics = compute_metrics_for_language(source, language)
        result["complexity"] = {
            "cyclomatic": metrics.cyclomatic_complexity,
            "loc": metrics.lines.sloc,
            "functions": metrics.n_complex_functions + metrics.n_long_functions,
            "maintainability": round(metrics.maintainability_index, 1),
        }
    except Exception as e:
        result["complexity"] = {"error": str(e)}

    # Readability (fast rule-based scoring)
    try:
        if registry.readability_scorer:
            r = registry.readability_scorer.score(source)
            result["readability"] = r.to_dict() if r else None
    except Exception as e:
        result["readability"] = {"error": str(e)}

    return result


@app.post("/analyze/multi-task", tags=["analysis"])
async def analyze_multi_task(req: AnalyzeRequest):
    """
    Run all 4 quality tasks through the shared multi-task encoder.
    Returns unified predictions for: security, complexity, bugs, patterns.
    Useful for ablation study comparison against 4 independent models.
    """
    if not registry.multi_task:
        raise HTTPException(503, "Multi-task model not loaded")
    source = req.code
    try:
        loop = asyncio.get_running_loop()
        pred = await loop.run_in_executor(
            _EXECUTOR, registry.multi_task.predict, source
        )
        return pred.to_dict()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Feedback endpoint — stores thumbs-up/down per analysis finding
# ---------------------------------------------------------------------------

_feedback_store: list[dict] = []  # seeded from disk on first feedback_stats call

class FeedbackRequest(BaseModel):
    analysis_id: Optional[str] = None
    finding_key: Optional[str] = Field(default="analysis_overall", description="Unique key for the finding or 'analysis_overall'")
    verdict: Literal["helpful", "not_helpful", "false_positive"] = Field(...)
    comment: Optional[str] = None

_FEEDBACK_PATH = Path("feedback.jsonl")


def _load_feedback_from_disk() -> list[dict]:
    if not _FEEDBACK_PATH.exists():
        return []
    entries = []
    try:
        with open(_FEEDBACK_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(_json.loads(line))
    except Exception:
        pass
    return entries


@app.post("/feedback", tags=["meta"])
def submit_feedback(req: FeedbackRequest):
    """Record user feedback on a specific analysis finding."""
    entry = {
        "analysis_id": req.analysis_id,
        "finding_key": req.finding_key,
        "verdict": req.verdict,
        "comment": req.comment,
        "ts": time.time(),
    }
    _feedback_store.append(entry)
    # Persist to disk so feedback survives restarts
    try:
        with open(_FEEDBACK_PATH, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Could not persist feedback to disk: %s", exc)
    return {"status": "ok", "stored": len(_feedback_store)}

@app.get("/feedback/stats", tags=["meta"])
def feedback_stats():
    """Returns aggregate feedback statistics."""
    if not _feedback_store:
        return {"total": 0, "helpful": 0, "not_helpful": 0, "false_positive": 0}
    counts = Counter(e["verdict"] for e in _feedback_store)
    return {
        "total": len(_feedback_store),
        "helpful": counts.get("helpful", 0),
        "not_helpful": counts.get("not_helpful", 0),
        "false_positive": counts.get("false_positive", 0),
    }


# ---------------------------------------------------------------------------
# GitHub OAuth & Webhook
# ---------------------------------------------------------------------------

@app.get("/auth/github", tags=["github"])
async def github_auth_redirect():
    """Redirect browser to GitHub OAuth authorization page."""
    client_id = os.environ.get("GITHUB_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth not configured — set GITHUB_CLIENT_ID env var",
        )
    redirect_uri = os.environ.get("GITHUB_REDIRECT_URI")
    if not redirect_uri:
        raise HTTPException(status_code=503, detail="GITHUB_REDIRECT_URI env var not configured")
    params = urllib.parse.urlencode(
        {"client_id": client_id, "scope": "repo read:user", "redirect_uri": redirect_uri}
    )
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")


@app.get("/auth/github/callback", tags=["github"])
async def github_auth_callback(code: str):
    """Exchange GitHub OAuth code for an access token, redirect to frontend."""
    client_id = os.environ.get("GITHUB_CLIENT_ID", "")
    client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8080")

    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="GitHub OAuth not configured")

    post_data = urllib.parse.urlencode(
        {"client_id": client_id, "client_secret": client_secret, "code": code}
    ).encode()
    req = urllib.request.Request(
        "https://github.com/login/oauth/access_token",
        data=post_data,
        headers={"Accept": "application/json"},
    )
    def _exchange() -> dict:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return _json.loads(resp.read())
    try:
        token_data = await asyncio.to_thread(_exchange)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub token exchange failed: {e}")

    access_token = token_data.get("access_token", "")
    if not access_token:
        error = token_data.get("error_description", "unknown error")
        raise HTTPException(status_code=400, detail=f"GitHub denied token: {error}")

    # Pass token via query param — frontend reads and cleans it immediately
    return RedirectResponse(f"{frontend_url}/?github_token={access_token}")


@app.post("/webhook/github", tags=["github"])
async def github_webhook(request: Request):
    """Receive GitHub push/PR webhooks. Verifies HMAC signature when GITHUB_WEBHOOK_SECRET is set."""
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise HTTPException(status_code=400, detail="Webhook not configured — set GITHUB_WEBHOOK_SECRET env var")

    body = await request.body()
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        webhook_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig_header, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = request.headers.get("X-GitHub-Event", "ping")
    if event == "ping":
        return {"status": "pong"}

    try:
        payload = _json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if event != "push":
        return {"status": "ignored", "event": event}

    repo_info = payload.get("repository", {})
    ref = payload.get("ref", "")
    branch = ref.split("/")[-1] if "/" in ref else ref
    commits = payload.get("commits", [])

    changed: list[str] = []
    for commit in commits:
        changed.extend(commit.get("added", []))
        changed.extend(commit.get("modified", []))

    EXT_RE = re.compile(r"\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|cs|rb|php|kt|swift)$")
    src_files = [f for f in changed if EXT_RE.search(f)][:20]

    return {
        "status": "received",
        "repo": repo_info.get("full_name", ""),
        "branch": branch,
        "changed_files": len(changed),
        "analyzable_files": len(src_files),
        "files": src_files,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
