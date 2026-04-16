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
    POST /analyze/bugs/diff
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

# Block HuggingFace network calls at startup — prevents model download hangs on Render
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA init on CPU servers

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
from pydantic import BaseModel, Field, field_validator

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
from models.cross_task_meta_learner import CrossTaskMetaLearner
from models.function_risk_localizer import FunctionRiskLocalizer
from models.complexity_trajectory import ComplexityTrajectoryPredictor
from models.code_grammar_anomaly import CodeGrammarAnomalyModel
from models.asymmetric_complexity_regressor import AsymmetricComplexityRegressor
from models.differential_risk_encoder import DifferentialRiskEncoder
from features.ood_detector import OODDetector, make_abstention_prediction
from features.code_metrics import metrics_to_feature_vector
from features.conformal_predictor import ResidualConformalPredictor


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
    meta_learner: CrossTaskMetaLearner | None = None
    function_localizer: FunctionRiskLocalizer | None = None
    trajectory_predictor: ComplexityTrajectoryPredictor | None = None
    grammar_anomaly: CodeGrammarAnomalyModel | None = None
    apcr: AsymmetricComplexityRegressor | None = None
    dre: DifferentialRiskEncoder | None = None
    ood_security: OODDetector | None = None   # OOD detector fitted on security training features
    ood_bug: OODDetector | None = None        # OOD detector fitted on bug predictor training features
    conformal_security: ResidualConformalPredictor | None = None
    conformal_bug: ResidualConformalPredictor | None = None
    conformal_complexity: ResidualConformalPredictor | None = None
    load_errors: dict[str, str] = {}


registry = ModelRegistry()
logger = logging.getLogger(__name__)

# Thread-pool for parallel CPU-bound model inference (one worker per logical CPU,
# capped at 8 to avoid excessive context-switching on smaller machines)
_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))

# ---------------------------------------------------------------------------
# Simple in-memory response cache (hash of code+filename → response dict)
# ---------------------------------------------------------------------------
_analysis_cache: dict[str, dict] = {}   # key → {"data": ..., "ts": float}
_specialist_cache: dict[str, dict] = {} # key → {"data": ..., "ts": float}
_CACHE_TTL_S = 86_400  # 24 hours — stale after model update

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


def _cache_key(code: str, filename: str, git_metadata=None) -> str:
    meta_dict = git_metadata.model_dump(exclude_none=True) if git_metadata else {}
    payload = f"{filename}\x00{code}\x00{str(sorted(meta_dict.items()))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _specialist_key(endpoint: str, code: str) -> str:
    return hashlib.sha256(f"{endpoint}\x00{code}".encode()).hexdigest()


def _cache_put(store: dict, key: str, value: dict, max_size: int = 200) -> None:
    if len(store) >= max_size:
        store.pop(next(iter(store)))
    store[key] = {"data": value, "ts": time.time()}


def _cache_get(store: dict, key: str) -> dict | None:
    """Return cached data if present and not expired; evict and return None if stale."""
    entry = store.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL_S:
        store.pop(key, None)
        return None
    return entry["data"]


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
    try:
        registry.clone_detector = CodeCloneDetector(enable_type4=False)
        logger.info("  [OK] Code clone detector")
    except Exception as e:
        registry.load_errors["clone_detector"] = str(e)
        logger.warning("  [WARN] Clone detector: %s", e)

    try:
        registry.refactoring_suggester = RefactoringSuggester()
        logger.info("  [OK] Refactoring suggester")
    except Exception as e:
        registry.load_errors["refactoring"] = str(e)
        logger.warning("  [WARN] Refactoring suggester: %s", e)

    try:
        registry.dead_code_detector = DeadCodeDetector()
        logger.info("  [OK] Dead code detector")
    except Exception as e:
        registry.load_errors["dead_code"] = str(e)
        logger.warning("  [WARN] Dead code detector: %s", e)

    try:
        registry.debt_estimator = TechnicalDebtEstimator()
        logger.info("  [OK] Technical debt estimator")
    except Exception as e:
        registry.load_errors["debt"] = str(e)
        logger.warning("  [WARN] Technical debt estimator: %s", e)

    try:
        registry.doc_analyzer = DocQualityAnalyzer()
        logger.info("  [OK] Documentation quality analyzer")
    except Exception as e:
        registry.load_errors["docs"] = str(e)
        logger.warning("  [WARN] Doc quality analyzer: %s", e)

    try:
        registry.performance_analyzer = PerformanceAnalyzer()
        logger.info("  [OK] Performance hotspot analyzer")
    except Exception as e:
        registry.load_errors["performance"] = str(e)
        logger.warning("  [WARN] Performance analyzer: %s", e)

    try:
        registry.dependency_analyzer = DependencyAnalyzer()
        logger.info("  [OK] Dependency & coupling analyzer")
    except Exception as e:
        registry.load_errors["dependencies"] = str(e)
        logger.warning("  [WARN] Dependency analyzer: %s", e)

    try:
        registry.readability_scorer = ReadabilityScorer()
        logger.info("  [OK] Code readability scorer")
    except Exception as e:
        registry.load_errors["readability"] = str(e)
        logger.warning("  [WARN] Readability scorer: %s", e)

    try:
        registry.multi_task = MultiTaskCodeModel()
        if registry.multi_task.ready:
            logger.info("  [OK] Multi-task model (mode=%s)", registry.multi_task.mode)
        else:
            logger.info("  [OK] Multi-task model (no checkpoint — fallback mode)")
    except Exception as e:
        logger.warning("  [WARN] Multi-task model failed to load: %s", e)
        registry.multi_task = None

    # Novel models: CTSL, FLRL, CTP
    try:
        registry.meta_learner = CrossTaskMetaLearner()
        if registry.meta_learner.load():
            logger.info("  [OK] Cross-task meta-learner (CTSL)")
        else:
            logger.info("  [--] Cross-task meta-learner — no checkpoint, using heuristic fallback")
    except Exception as e:
        logger.warning("  [WARN] Cross-task meta-learner: %s", e)
        registry.meta_learner = None

    try:
        registry.function_localizer = FunctionRiskLocalizer(top_k=5)
        logger.info("  [OK] Function-level risk localizer (FLRL)")
    except Exception as e:
        logger.warning("  [WARN] Function risk localizer: %s", e)
        registry.function_localizer = None

    try:
        registry.trajectory_predictor = ComplexityTrajectoryPredictor()
        logger.info("  [OK] Complexity trajectory predictor (CTP)")
    except Exception as e:
        logger.warning("  [WARN] Complexity trajectory predictor: %s", e)
        registry.trajectory_predictor = None

    try:
        registry.grammar_anomaly = CodeGrammarAnomalyModel()
        if registry.grammar_anomaly.load("checkpoints/cgam"):
            logger.info("  [OK] Code Grammar Anomaly Model (CGAM)")
        else:
            logger.info("  [--] CGAM — no checkpoint, predict() returns null score until trained")
    except Exception as e:
        logger.warning("  [WARN] CGAM: %s", e)
        registry.grammar_anomaly = None

    try:
        registry.apcr = AsymmetricComplexityRegressor()
        if registry.apcr.load("checkpoints/apcr"):
            logger.info("  [OK] Asymmetric Complexity Regressor (APCR)")
        else:
            logger.info("  [--] APCR — no checkpoint, will use heuristic fallback")
    except Exception as e:
        logger.warning("  [WARN] APCR: %s", e)
        registry.apcr = None

    try:
        registry.dre = DifferentialRiskEncoder()
        if registry.dre.load("checkpoints/dre"):
            logger.info("  [OK] Differential Risk Encoder (DRE)")
        else:
            logger.info("  [--] DRE — no checkpoint, will use heuristic fallback")
    except Exception as e:
        logger.warning("  [WARN] DRE: %s", e)
        registry.dre = None

    # OOD detectors — fitted once during training; loaded here for inference
    try:
        registry.ood_security = OODDetector.load("checkpoints/security/ood_detector.pkl")
        if registry.ood_security:
            logger.info("  [OK] OOD detector (security)")
        else:
            logger.info("  [--] OOD detector (security) — checkpoint not found, OOD checks disabled")
    except Exception as e:
        logger.warning("  [WARN] OOD detector (security): %s", e)

    try:
        registry.ood_bug = OODDetector.load("checkpoints/bug_predictor/ood_detector.pkl")
        if registry.ood_bug:
            logger.info("  [OK] OOD detector (bug predictor)")
        else:
            logger.info("  [--] OOD detector (bug predictor) — checkpoint not found, OOD checks disabled")
    except Exception as e:
        logger.warning("  [WARN] OOD detector (bug predictor): %s", e)

    # Conformal predictors — load calibrated quantiles (fall back to defaults if not found)
    registry.conformal_security = ResidualConformalPredictor.load(
        "checkpoints/security/conformal.json", task="security"
    )
    registry.conformal_bug = ResidualConformalPredictor.load(
        "checkpoints/bug_predictor/conformal.json", task="bug"
    )
    registry.conformal_complexity = ResidualConformalPredictor.load(
        "checkpoints/complexity/conformal.json", task="complexity"
    )
    logger.info(
        "  [OK] Conformal predictors loaded (security fallback=%s, bug fallback=%s, complexity fallback=%s)",
        registry.conformal_security._is_fallback,
        registry.conformal_bug._is_fallback,
        registry.conformal_complexity._is_fallback,
    )

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

_extra_origin = os.environ.get("FRONTEND_URL", "").strip()
_allowed_origins = [
    "https://safaapatel.github.io",
    "http://localhost:5173",
    "http://localhost:4173",
]
if _extra_origin:
    _allowed_origins.append(_extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Internal-Key", "Accept", "Authorization"],
)

# Optional API key guard — only active when INTELLICODE_API_KEY env var is set.
# /health is exempt for uptime monitors. /docs and /openapi.json are NOT exempt —
# they reveal the full API schema and should be protected in production.
_API_KEY_SECRET = os.environ.get("INTELLICODE_API_KEY", "").strip()
_API_KEY_EXEMPT = {"/health"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY_SECRET and request.url.path not in _API_KEY_EXEMPT:
        # Accept key via header only — never via query param (query params appear in logs)
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, _API_KEY_SECRET):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Attach security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # Only add HSTS when serving over HTTPS (Render/production)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Apply sliding-window rate limiting to all /analyze/* routes.
    Limits are per client IP: 20 requests/minute by default.
    /health, /models, /stats, and /docs are exempt.
    """
    _RATE_EXEMPT_PREFIXES = ("/health", "/models", "/stats", "/docs", "/openapi", "/redoc")
    path = request.url.path
    if not any(path.startswith(p) for p in _RATE_EXEMPT_PREFIXES):
        try:
            _check_rate_limit(request)
        except HTTPException as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers={"Retry-After": "60"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

_SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "java", "go", "rust",
    "cpp", "c", "csharp", "ruby", "php", "kotlin", "swift", "unknown",
}


class GitMetadataInput(BaseModel):
    """Validated git metadata for bug prediction. All fields are optional."""
    code_churn: Optional[int] = Field(default=None, ge=0, le=1_000_000)
    author_count: Optional[int] = Field(default=None, ge=1, le=10_000)
    file_age_days: Optional[int] = Field(default=None, ge=0, le=36_500)
    n_past_bugs: Optional[int] = Field(default=None, ge=0, le=10_000)
    commit_freq: Optional[float] = Field(default=None, ge=0.0, le=1_000.0)


_MAX_SLOC = 2000   # lines beyond this get truncated with a warning in the response

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze", min_length=1, max_length=500_000)
    filename: str = Field(default="snippet.py", description="Filename for context")
    language: str = Field(default="python", description="Programming language")
    git_metadata: Optional[GitMetadataInput] = Field(
        default=None,
        description=(
            "Optional git metadata: {code_churn, author_count, "
            "file_age_days, n_past_bugs, commit_freq}"
        ),
    )

    @field_validator("code")
    @classmethod
    def guard_input_length(cls, v: str) -> str:
        lines = v.splitlines()
        if len(lines) > _MAX_SLOC:
            # Truncate rather than reject: analysis still runs on the first N lines.
            # The truncation flag is surfaced in the response (see _maybe_truncate helper).
            return "\n".join(lines[:_MAX_SLOC])
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in _SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. Supported: {', '.join(sorted(_SUPPORTED_LANGUAGES))}"
            )
        return normalized


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
    abstained: bool = False
    low_confidence: bool = False
    low_confidence_reason: Optional[str] = None
    probability_adjusted: Optional[float] = None
    top_feature_importances: list[dict] = Field(default_factory=list)
    reliability_context: dict = Field(default_factory=dict)


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
    # Novel model results (always included, may be null if untrained)
    grammar_anomaly: Optional[dict] = None
    apcr: Optional[dict] = None
    function_risk: Optional[dict] = None


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
            "ood_detector_security": "ready" if registry.ood_security else "unavailable",
            "ood_detector_bug": "ready" if registry.ood_bug else "unavailable",
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


@app.get("/models/evaluation", tags=["meta"])
def models_evaluation():
    """
    Cross-project generalisation evaluation results (LOPO benchmarks).

    Returns per-task LOPO results loaded from evaluation/results/lopo_*.json.
    Also includes the temporal split degradation result for bug prediction
    (random-split AUC vs temporal-split AUC).
    """
    results_dir = Path("evaluation/results")

    def _load(name: str) -> dict:
        p = results_dir / name
        if not p.exists():
            return {}
        try:
            return json.load(open(p))
        except Exception:
            return {}

    security  = _load("lopo_security.json")
    bug       = _load("lopo_bug.json")
    complexity = _load("lopo_complexity.json")

    # Temporal split results for bug prediction (from baseline_comparison.json)
    temporal_bug: dict = {}
    baseline = _load("baseline_comparison.json")
    if baseline:
        bugs_eval = baseline.get("bugs", {})
        halstead_random = bugs_eval.get("halstead_lr", {})
        halstead_temporal = bugs_eval.get("halstead_lr_temporal", {})
        temporal_bug = {
            "random_split_auc":   halstead_random.get("auc"),
            "temporal_split_auc": halstead_temporal.get("auc"),
            "delta_auc":          halstead_temporal.get("vs_random_split_auc_delta"),
            "protocol":           halstead_temporal.get("protocol"),
            "n_train":            halstead_temporal.get("n_train"),
            "n_test":             halstead_temporal.get("n_test"),
            "note": (
                "Temporal split trains on older commits and tests on newer ones, "
                "preventing SZZ-style label leakage. The AUC drop from random to "
                "temporal split quantifies overestimation from data leakage."
            ),
        }

    return {
        "security":    security,
        "bug":         bug,
        "complexity":  complexity,
        "temporal_bug": temporal_bug,
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


def _parse_git_meta(raw: Optional["GitMetadataInput"]) -> Optional[GitMetadata]:
    """Parse optional GitMetadataInput into a GitMetadata dataclass."""
    if not raw:
        return None
    d = raw.model_dump(exclude_none=True)
    return GitMetadata(**{
        k: d.get(k, 0)
        for k in ("code_churn", "author_count", "file_age_days", "n_past_bugs", "commit_freq")
    })


# ---------------------------------------------------------------------------
# Decision layer — converts ML outputs into actionable guidance
# ---------------------------------------------------------------------------

# Languages the security model was validated on (Python-primary training data)
_SECURITY_RELIABLE_LANGUAGES = {"python"}
# Languages complexity model is reliable on (calibrated against SonarQube)
_COMPLEXITY_RELIABLE_LANGUAGES = {"python", "javascript", "typescript", "java", "go"}

# SLOC range within training distribution
_TRAINING_SLOC_MIN = 5
_TRAINING_SLOC_MAX = 600


def _is_ood(language: str, sloc: int | None = None, model: str = "security") -> bool:
    """
    Return True if this code is likely out-of-distribution for the given model.
    Uses language coverage and SLOC range as proxies for training distribution.
    """
    lang = (language or "python").lower()
    if model == "security" and lang not in _SECURITY_RELIABLE_LANGUAGES:
        return True
    if model == "complexity" and lang not in _COMPLEXITY_RELIABLE_LANGUAGES:
        return True
    if sloc is not None:
        if sloc < _TRAINING_SLOC_MIN or sloc > _TRAINING_SLOC_MAX:
            return True
    return False


def _compute_decision(
    severity: str,
    confidence: float,
    is_ood: bool,
    model: str = "security",
) -> dict:
    """
    Convert a raw finding into an actionable decision with user-facing language.

    Returns:
        action:      "fix_now" | "review_manually" | "low_priority" | "unreliable"
        label:       Short label for UI badge
        explanation: One-sentence explanation in plain language
        priority:    int 1–4 (lower = more urgent)
        color:       Tailwind color token for UI
    """
    if is_ood:
        return {
            "action": "unreliable",
            "label": "Low reliability",
            "explanation": (
                f"This finding was generated by a model not validated on this codebase type. "
                f"Treat as a weak signal — human review recommended."
            ),
            "priority": 4,
            "color": "gray",
        }

    sev = severity.lower()

    # Bug predictor: temporal AUC = 0.46 — cap confidence and always recommend review
    if model == "bug":
        if sev in ("critical", "high"):
            return {
                "action": "review_manually",
                "label": "Review manually",
                "explanation": (
                    "Bug risk is elevated. Note: this model reflects historical defect patterns "
                    "in similar codebases, not a direct measurement of defects in this file."
                ),
                "priority": 2,
                "color": "yellow",
            }
        return {
            "action": "low_priority",
            "label": "Low priority",
            "explanation": "Low historical defect signal. Monitor during next review cycle.",
            "priority": 3,
            "color": "blue",
        }

    # Security / pattern findings
    if sev in ("critical", "high") and confidence >= 0.75:
        return {
            "action": "fix_now",
            "label": "Fix immediately",
            "explanation": (
                f"High severity and high model confidence ({confidence:.0%}). "
                f"This class of issue is reliably detected in similar Python codebases."
            ),
            "priority": 1,
            "color": "red",
        }

    if sev in ("critical", "high") and confidence < 0.75:
        return {
            "action": "review_manually",
            "label": "Review manually",
            "explanation": (
                f"Potentially serious, but model confidence is limited ({confidence:.0%}). "
                f"Verify before acting — may be a false positive."
            ),
            "priority": 2,
            "color": "yellow",
        }

    return {
        "action": "low_priority",
        "label": "Low priority",
        "explanation": "Minor issue. Address during routine maintenance.",
        "priority": 3,
        "color": "blue",
    }


def _attach_decisions(security_out: dict, language: str, sloc: int | None) -> dict:
    """
    Mutate security findings in-place to add a 'decision' field to each finding.
    Returns the mutated dict.
    """
    lang = (language or "python").lower()
    ood = _is_ood(lang, sloc, model="security")
    findings = security_out.get("findings", [])
    for f in findings:
        f["decision"] = _compute_decision(
            severity=f.get("severity", "low"),
            confidence=f.get("confidence", 0.5),
            is_ood=ood,
            model="security",
        )
    return security_out


def _build_trust_summary(
    language: str,
    sloc: int | None,
    security_out: dict,
    bug_out,
) -> dict:
    """
    Build a top-level trust summary that tells the user how reliable each
    model's output is for this specific code submission.
    """
    lang = (language or "python").lower()
    sec_ood = _is_ood(lang, sloc, model="security")
    comp_ood = _is_ood(lang, sloc, model="complexity")

    crit_findings = [
        f for f in security_out.get("findings", [])
        if f.get("severity") == "critical"
    ]
    high_conf_crits = [f for f in crit_findings if f.get("confidence", 0) >= 0.75]

    items = []

    if sec_ood:
        items.append({
            "model": "Security",
            "reliability": "low",
            "message": (
                f"Security model not validated on {lang} codebases. "
                f"Findings are indicative only — manual review recommended."
            ),
        })
    elif crit_findings and not high_conf_crits:
        items.append({
            "model": "Security",
            "reliability": "medium",
            "message": (
                "Critical findings detected but model confidence is below 75%. "
                "Review carefully before treating as confirmed vulnerabilities."
            ),
        })
    else:
        items.append({
            "model": "Security",
            "reliability": "high" if not sec_ood else "low",
            "message": (
                "Security analysis validated on Python codebases (in-dist AUC=0.83). "
                "High-confidence findings are likely real."
            ) if not sec_ood else "Low reliability — out of training distribution.",
        })

    if comp_ood:
        items.append({
            "model": "Complexity",
            "reliability": "medium",
            "message": (
                f"Complexity model is most accurate on Python (r=0.98 vs SonarQube). "
                f"Results for {lang} are directionally correct but less calibrated."
            ),
        })
    else:
        items.append({
            "model": "Complexity",
            "reliability": "high",
            "message": "Complexity scores calibrated against SonarQube (Pearson r=0.98).",
        })

    items.append({
        "model": "Bug Prediction",
        "reliability": "low",
        "message": (
            "Bug probability reflects historical defect patterns in similar codebases, "
            "not a direct scan for bugs in this file. "
            "Use as a risk indicator, not a defect count."
        ),
    })

    overall_reliable = not sec_ood and not comp_ood
    return {
        "overall_reliable": overall_reliable,
        "language_in_distribution": lang in _SECURITY_RELIABLE_LANGUAGES,
        "items": items,
    }


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
    cached = _cache_get(_analysis_cache, cache_key)
    if cached is not None:
        return cached

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

    # --- C1 fix: bump bug probability when dangerous security patterns are found ---
    # The bug ML model is trained on git/complexity features and underestimates risk
    # for code that contains dangerous API calls (eval, pickle, shell=True, etc.)
    if bug_out is not None:
        crit_sec = security_out.get("summary", {}).get("critical", 0)
        high_sec = security_out.get("summary", {}).get("high", 0)
        dangerous_types = {f.get("vuln_type", "") for f in security_out.get("findings", [])}
        high_danger = dangerous_types & {"command_injection", "insecure_deserialization", "code_injection"}
        if crit_sec >= 2 or high_danger:
            # Inject a floor: these patterns strongly predict real bugs/incidents
            floor = 0.45 if crit_sec >= 3 else 0.30
            if bug_out.bug_probability < floor:
                from dataclasses import replace as _dc_replace
                bug_out = _dc_replace(
                    bug_out,
                    bug_probability=round(floor, 3),
                    static_score=round(floor, 3),
                    risk_level="medium" if floor < 0.55 else "high",
                    risk_factors=list(bug_out.risk_factors) + [
                        f"Security findings ({crit_sec} critical, {high_sec} high) indicate elevated bug risk"
                    ],
                )

    # --- Wave 2: debt (needs wave-1 results) + specialists + novel models (independent) ---
    (
        (debt_out, debt_result),
        docs_out,
        perf_out,
        deps_out,
        read_out,
        grammar_out,
        apcr_out,
        func_risk_out,
    ) = await asyncio.gather(
        _run(functools.partial(
            _step_debt, source, security_out,
            complexity_result, bug_result, clone_result, dead_result, refactor_result,
        )),
        _run(_step_docs, source),
        _run(_step_performance, source),
        _run(_step_dependencies, source),
        _run(_step_readability, source),
        _run(_step_grammar_anomaly, source),
        _run(_step_apcr, source),
        _run(_step_function_risk, source),
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
        f"Bug risk: {bug_out.risk_level if bug_out is not None else 'unknown'}. "
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
        grammar_anomaly=grammar_out or None,
        apcr=apcr_out or None,
        function_risk=func_risk_out or None,
    )
    _cache_put(_analysis_cache, cache_key, response.model_dump())
    return response


@app.post("/analyze/security", tags=["analysis"])
def analyze_security(req: AnalyzeRequest):
    """Security vulnerability scan only."""
    t_start = time.perf_counter()

    language = detect_language(req.filename, req.language)
    if registry.security and language == "python":
        ml_disabled = getattr(registry.security, "_ml_disabled", False)
        findings = registry.security.predict(req.code)
        vuln_score = registry.security.vulnerability_score(req.code)
        result = {
            "findings": [vars(f) for f in findings],
            "vulnerability_score": round(vuln_score, 3),
            # Transparency: tell the client which inference path was taken.
            # ml_disabled=True means the ML ensemble is off (LOPO AUC below threshold)
            # and the score comes from the pattern scanner only.
            "ml_disabled": ml_disabled,
            "ml_source": (
                "pattern_scanner_only" if ml_disabled
                else "rf_cnn_contrastive_ensemble" if getattr(registry.security, "_contrastive_ready", False)
                else "rf_cnn_ensemble"
            ),
        }

        # OOD check for security
        try:
            import numpy as _np
            from features.code_metrics import compute_all_metrics as _cam
            _metrics = _cam(req.code)
            _feat = _np.array(metrics_to_feature_vector(_metrics))
            ood_meta = make_abstention_prediction(
                vuln_score, _feat, registry.ood_security, task="security"
            )
            result["ood"] = {k: v for k, v in ood_meta.items() if k != "probability"}
            if ood_meta.get("low_confidence"):
                result["low_confidence"] = True
                result["low_confidence_reason"] = (
                    f"Input is {ood_meta.get('sigma_distance', 0):.1f} sigma from training "
                    "distribution — vulnerability score may not be reliable."
                )
        except Exception:
            pass

        # Conformal prediction interval for vulnerability_score
        if registry.conformal_security:
            result["conformal_interval"] = registry.conformal_security.to_dict(vuln_score)
    else:
        result = scan_security_patterns_for_language(req.code, language).to_dict()
        result["ml_disabled"] = True
        result["ml_source"] = "pattern_scanner_only"
        # Conformal interval still available for pattern-based score
        if registry.conformal_security:
            score = result.get("vulnerability_score", 0.0)
            result["conformal_interval"] = registry.conformal_security.to_dict(float(score))

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

    # Conformal prediction interval for predicted cognitive complexity
    if registry.conformal_complexity:
        predicted_cc = float(out.get("predicted_complexity", out.get("cognitive_complexity", 0.0)))
        out["conformal_interval"] = registry.conformal_complexity.to_dict(predicted_cc)

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
    """Bug probability prediction from static file-level features."""
    t_start = time.perf_counter()

    git_meta = _parse_git_meta(req.git_metadata)
    if not registry.bug_predictor:
        raise HTTPException(status_code=503, detail="Bug prediction model unavailable")

    result = registry.bug_predictor.predict(req.code, git_meta)
    out = result.to_dict()

    # OOD check
    try:
        from features.code_metrics import compute_all_metrics
        import numpy as np
        metrics = compute_all_metrics(req.code)
        feat_vec = metrics_to_feature_vector(metrics)
        ood_meta = make_abstention_prediction(
            float(out.get("bug_probability", 0.0)),
            np.array(feat_vec),
            registry.ood_bug,
            task="bug",
        )
        out["ood"] = {k: v for k, v in ood_meta.items() if k != "probability"}
        if ood_meta.get("low_confidence"):
            out["low_confidence"] = True
            out["low_confidence_reason"] = (
                f"Input is {ood_meta.get('sigma_distance', 0):.1f} sigma from training "
                "distribution — bug probability estimate may not be reliable."
            )
            out["probability_adjusted"] = ood_meta.get("probability")
    except Exception:
        pass

    # Conformal prediction interval for bug_probability
    if registry.conformal_bug:
        prob = float(out.get("bug_probability", out.get("probability", 0.0)))
        out["conformal_interval"] = registry.conformal_bug.to_dict(prob)

    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


class DiffBugRequest(BaseModel):
    """Request body for diff-aware bug prediction."""
    diff: str = Field(..., min_length=1, max_length=500_000, description="Unified diff text (git diff output)")
    filename: str = Field("unknown", description="Primary file being changed")


@app.post("/analyze/bugs/diff", tags=["analysis"])
def analyze_bugs_diff(req: DiffBugRequest):
    """
    Diff-aware bug prediction — operates on a unified diff rather than a static snapshot.

    This endpoint uses commit-level features (added/deleted lines, hunk count,
    identifier semantics) rather than file-level metrics, matching the Kamei et al.
    JIT defect prediction approach. It avoids the temporal leakage problem of the
    static file-level model.

    Body: { "diff": "<unified diff text>", "filename": "path/to/file.py" }
    """
    t_start = time.perf_counter()
    try:
        from models.diff_bug_predictor import DiffBugPredictor
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Diff bug predictor unavailable: {e}")

    try:
        predictor = DiffBugPredictor()
        probability = predictor.predict_proba(req.diff)
        risky_hunks = predictor.top_risky_hunks(req.diff, top_k=5)
        risk_label = (
            "high" if probability >= 0.7
            else "medium" if probability >= 0.4
            else "low"
        )
        out = {
            "probability": round(probability, 4),
            "risk_label": risk_label,
            "model": "diff_bug_predictor",
            "risky_hunks": risky_hunks,
            "note": (
                "Diff-level prediction avoids temporal leakage. "
                "Use this endpoint for commit-level risk assessment in CI."
            ),
            "duration_seconds": round(time.perf_counter() - t_start, 3),
        }
        return out
    except Exception as e:
        logger.exception("Diff bug predictor error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/clones", tags=["analysis"])
def analyze_clones(req: AnalyzeRequest):
    """Code clone detection — finds Type-1, Type-2, and Type-3 duplicates."""
    if not registry.clone_detector:
        raise HTTPException(status_code=503, detail="Clone detector unavailable")
    t_start = time.perf_counter()
    result = registry.clone_detector.detect(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/refactoring", tags=["analysis"])
def analyze_refactoring(req: AnalyzeRequest):
    """Refactoring suggestions — actionable recommendations with effort estimates."""
    if not registry.refactoring_suggester:
        raise HTTPException(status_code=503, detail="Refactoring suggester unavailable")
    t_start = time.perf_counter()
    result = registry.refactoring_suggester.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/dead-code", tags=["analysis"])
def analyze_dead_code(req: AnalyzeRequest):
    """Dead code detection — unused imports, unreachable code, empty except blocks, etc."""
    if not registry.dead_code_detector:
        raise HTTPException(status_code=503, detail="Dead code detector unavailable")
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
    _cached_full = _cache_get(_analysis_cache, cache_key)
    if _cached_full is not None:
        cached = _cached_full
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
    clone_result = registry.clone_detector.detect(req.code) if registry.clone_detector else None
    dead_result = registry.dead_code_detector.detect(req.code) if registry.dead_code_detector else None
    refactor_result = registry.refactoring_suggester.analyze(req.code) if registry.refactoring_suggester else None

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
    if not registry.doc_analyzer:
        raise HTTPException(status_code=503, detail="Documentation analyzer unavailable")
    key = _specialist_key("docs", req.code)
    cached = _cache_get(_specialist_cache, key)
    if cached is not None:
        return cached
    t_start = time.perf_counter()
    result = registry.doc_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/performance", tags=["analysis"])
def analyze_performance(req: AnalyzeRequest):
    """Performance hotspot detection — finds O(n²) loops, I/O in loops, etc."""
    if not registry.performance_analyzer:
        raise HTTPException(status_code=503, detail="Performance analyzer unavailable")
    key = _specialist_key("performance", req.code)
    cached = _cache_get(_specialist_cache, key)
    if cached is not None:
        return cached
    t_start = time.perf_counter()
    result = registry.performance_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/dependencies", tags=["analysis"])
def analyze_dependencies(req: AnalyzeRequest):
    """Dependency & coupling analysis — fan-out, wildcard imports, coupling score."""
    if not registry.dependency_analyzer:
        raise HTTPException(status_code=503, detail="Dependency analyzer unavailable")
    key = _specialist_key("dependencies", req.code)
    cached = _cache_get(_specialist_cache, key)
    if cached is not None:
        return cached
    t_start = time.perf_counter()
    result = registry.dependency_analyzer.analyze(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    _cache_put(_specialist_cache, key, out)
    return out


@app.post("/analyze/readability", tags=["analysis"])
def analyze_readability(req: AnalyzeRequest):
    """Code readability scoring — naming, comments, structure, cognitive load."""
    if not registry.readability_scorer:
        raise HTTPException(status_code=503, detail="Readability scorer unavailable")
    key = _specialist_key("readability", req.code)
    cached = _cache_get(_specialist_cache, key)
    if cached is not None:
        return cached
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
        ml_disabled = getattr(registry.security, "_ml_disabled", False)

        # OOD check: compute feature vector and test against training distribution
        ood_meta: dict = {}
        try:
            from features.code_metrics import compute_all_metrics
            metrics = compute_all_metrics(source)
            feat_vec = metrics_to_feature_vector(metrics)
            import numpy as np
            ood_meta = make_abstention_prediction(0.0, np.array(feat_vec), registry.ood_security, task="security")
            ood_meta.pop("probability", None)  # remove irrelevant field
        except Exception:
            pass

        findings = registry.security.predict(source)
        sec_score = registry.security.vulnerability_score(source)

        result = {
            "findings": [vars(f) for f in findings],
            "vulnerability_score": round(sec_score, 3),
            "ml_disabled": ml_disabled,
            "ml_source": "pattern_scanner_only" if ml_disabled else "rf_cnn_ensemble",
            "summary": {
                "total": len(findings),
                "critical": sum(1 for f in findings if f.severity == "critical"),
                "high": sum(1 for f in findings if f.severity == "high"),
                "medium": sum(1 for f in findings if f.severity == "medium"),
                "low": sum(1 for f in findings if f.severity == "low"),
            },
        }
        if ood_meta:
            result["ood"] = ood_meta
            if ood_meta.get("low_confidence"):
                result["low_confidence"] = True
                result["low_confidence_reason"] = (
                    f"Input is {ood_meta.get('sigma_distance', 0):.1f} sigma from training "
                    f"distribution — predictions may be unreliable for this code pattern."
                )
        return result
    result = scan_security_patterns_for_language(source, language).to_dict()
    result["ml_disabled"] = True
    result["ml_source"] = "pattern_scanner_only"
    return result


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
    out_dict = r.to_dict()

    # OOD check: flag inputs far from the bug predictor's training distribution
    try:
        from features.code_metrics import compute_all_metrics
        import numpy as np
        metrics = compute_all_metrics(source)
        feat_vec = metrics_to_feature_vector(metrics)
        ood_meta = make_abstention_prediction(
            float(out_dict.get("bug_probability", 0.0)),
            np.array(feat_vec),
            registry.ood_bug,
            task="bug",
        )
        if ood_meta.get("low_confidence"):
            out_dict["low_confidence"] = True
            out_dict["low_confidence_reason"] = (
                f"Input is {ood_meta.get('sigma_distance', 0):.1f} sigma from training "
                f"distribution — bug probability estimate may not be reliable."
            )
            # Surface the adjusted (scaled-down) probability alongside the raw value
            out_dict["probability_adjusted"] = ood_meta.get("probability")
        out_dict["ood"] = {k: v for k, v in ood_meta.items() if k != "probability"}
    except Exception:
        pass

    return BugPredictionOut(**out_dict), r


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


def _step_grammar_anomaly(source: str) -> dict:
    """Run CGAM — returns empty dict if untrained."""
    if not registry.grammar_anomaly or not registry.grammar_anomaly.ready:
        return {}
    try:
        return registry.grammar_anomaly.predict(source).to_dict()
    except Exception as e:
        logger.warning("CGAM predict failed: %s", e)
        return {}


def _step_apcr(source: str) -> dict:
    """Run APCR — uses heuristic fallback if untrained."""
    if not registry.apcr:
        return {}
    try:
        import numpy as np
        metrics = compute_all_metrics(source)
        feat_vec = metrics_to_feature_vector(metrics)
        return registry.apcr.predict(np.asarray(feat_vec, dtype=np.float32)).to_dict()
    except Exception as e:
        logger.warning("APCR predict failed: %s", e)
        return {}


def _step_function_risk(source: str) -> dict:
    """Run FLRL function-level risk localizer."""
    if not registry.function_localizer:
        return {}
    try:
        return registry.function_localizer.localize(source).to_dict()
    except Exception as e:
        logger.warning("FLRL predict failed: %s", e)
        return {}


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
        cached = _cache_get(_analysis_cache, cache_key)
        if cached is not None:
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
            f"Bug risk: {bug_out.risk_level if bug_out is not None else 'unknown'}. "
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

        # --- Decision layer: attach per-finding decisions + trust summary ---
        # Estimate SLOC from source line count (fast proxy, no extra compute)
        sloc_val = len([l for l in source.splitlines() if l.strip() and not l.strip().startswith("#")])
        if response_dict.get("security"):
            _attach_decisions(response_dict["security"], language, sloc_val)
        # Attach decision to bug prediction output
        if bug_out is not None and response_dict.get("bug_prediction"):
            bug_decision = _compute_decision(
                severity=bug_out.risk_level,
                confidence=getattr(bug_out, "confidence", 0.5),
                is_ood=_is_ood(language, sloc_val, model="bug"),
                model="bug",
            )
            response_dict["bug_prediction"]["decision"] = bug_decision
        response_dict["trust_summary"] = _build_trust_summary(
            language, sloc_val,
            response_dict.get("security", {}),
            bug_out,
        )
        response_dict["model_version"] = "1.1.0"

        _cache_put(_analysis_cache, cache_key, response_dict)
        yield evt({"status": "complete", "result": response_dict})

    async def safe_generate():
        """Wraps generate() so the SSE stream always ends with a terminal event."""
        try:
            async for chunk in generate():
                yield chunk
        except Exception as exc:
            logger.error("Stream analysis crashed: %s", exc, exc_info=True)
            yield f"data: {_json.dumps({'status': 'error', 'error': str(exc), 'message': 'Analysis failed unexpectedly — please retry.'})}\n\n"

    return StreamingResponse(
        safe_generate(),
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


@app.post("/analyze/meta-risk", tags=["analysis"])
def analyze_meta_risk(req: AnalyzeRequest):
    """
    Cross-Task Stacking Meta-Learner (CTSL) — unified risk score.

    Runs all four base models, constructs domain-interaction features
    (security*bug, complexity_deficit*bug, pattern_risk*bug, etc.) and
    produces a calibrated joint risk score that outperforms any single model.

    Returns unified_risk [0,1], per-task calibrated scores, the dominant
    contributing signal, and the top-3 interaction feature contributions.
    """
    if not registry.meta_learner:
        raise HTTPException(503, "Cross-task meta-learner unavailable")
    if not registry.bug_predictor or not registry.complexity or not registry.security:
        raise HTTPException(503, "Base models not loaded — cannot compute meta-risk")

    source = req.code
    t_start = time.perf_counter()

    bug_res = registry.bug_predictor.predict(source, None)
    cpx_res = registry.complexity.predict(source)
    findings = registry.security.predict(source)
    sec_score = registry.security.vulnerability_score(source)
    n_critical = sum(1 for f in findings if f.severity == "critical")

    pattern_label, pattern_conf = "clean", 0.5
    if registry.pattern_model and registry.pattern_model.ready:
        try:
            pat = registry.pattern_model.predict(source)
            pattern_label = pat.label
            pattern_conf  = pat.confidence
        except Exception:
            pass

    result = registry.meta_learner.predict(
        security_score=sec_score,
        complexity_score=float(cpx_res.score),
        bug_probability=float(bug_res.bug_probability),
        pattern_label=pattern_label,
        pattern_confidence=pattern_conf,
        n_security_findings=len(findings),
        n_critical=n_critical,
    )
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/function-risk", tags=["analysis"])
def analyze_function_risk(req: AnalyzeRequest):
    """
    Function-Level Risk Localizer (FLRL) — decomposes file risk to functions.

    Scores each function individually using complexity-weighted attribution
    and ranks functions by their attributed risk score (raw_score * share
    of file's total cognitive complexity).

    Returns a ranked list of functions with raw_score, complexity_weight,
    attributed_score, risk_level, and a plain-English reason.
    """
    if not registry.function_localizer:
        raise HTTPException(503, "Function risk localizer unavailable")

    t_start = time.perf_counter()
    result = registry.function_localizer.localize(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


class TrajectoryRequest(BaseModel):
    snapshots: list[tuple[float, float]] = Field(
        ...,
        description="List of [unix_timestamp, cognitive_complexity] pairs, oldest-first",
        min_length=2,
    )
    current_complexity: Optional[float] = Field(
        None, description="Override for current complexity (defaults to last snapshot)"
    )


@app.post("/analyze/trajectory", tags=["analysis"])
def analyze_trajectory(req: TrajectoryRequest):
    """
    Complexity Trajectory Predictor (CTP) — predicts complexity trend direction.

    Given a sequence of (timestamp, cognitive_complexity) snapshots from git
    history, fits a time-decay weighted linear regression and predicts whether
    complexity is increasing, stable, or decreasing.

    Returns slope, trajectory, risk_multiplier (for adjusting bug probability),
    30-day forecast, and fit confidence (R^2).

    Body: { "snapshots": [[ts, cog], ...], "current_complexity": 14.0 }
    """
    if not registry.trajectory_predictor:
        raise HTTPException(503, "Trajectory predictor unavailable")

    t_start = time.perf_counter()
    result = registry.trajectory_predictor.predict(
        snapshots=[(float(ts), float(cog)) for ts, cog in req.snapshots],
        current_complexity=req.current_complexity,
    )
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/grammar-anomaly", tags=["analysis"])
def analyze_grammar_anomaly(req: AnalyzeRequest):
    """
    Code Grammar Anomaly Model (CGAM) — detects anomalous code structure.

    Trains a Variable-Order Markov Model over DFS-order AST node-type
    sequences on unlabeled clean code, then flags files whose AST grammar
    has high perplexity under the learned model.

    Requires the CGAM checkpoint to be trained first (POST /train/cgam or
    via training/train_cgam.py).  Without a checkpoint, returns a null
    result with is_anomalous=false.

    Returns: perplexity, anomaly_score [0-1], is_anomalous, top_anomalous_ngrams,
             grammar_coverage (fraction of file's trigrams seen in training).
    """
    if not registry.grammar_anomaly:
        raise HTTPException(503, "Grammar Anomaly Model unavailable")

    t_start = time.perf_counter()
    result = registry.grammar_anomaly.predict(req.code)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 4)
    return out


class DRERequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=500_000)
    filename: str = "snippet.py"
    prev_features: Optional[list[float]] = Field(
        None,
        description="Static feature vector from the previous commit (15-dim). "
                    "If omitted, treated as zero vector (first commit).",
    )


@app.post("/analyze/apcr", tags=["analysis"])
def analyze_apcr(req: AnalyzeRequest):
    """
    Asymmetric Complexity Regressor (APCR) — predicts cognitive complexity
    with an asymmetric pinball loss that penalises underestimation 3x more
    than overestimation.

    Returns: prediction, lower/upper bounds, asymmetry_penalty (how much
    the asymmetric loss raised the estimate vs RMSE), risk_flag (>= 12),
    and model confidence.

    Falls back to cyclomatic complexity if no checkpoint is available.
    """
    if not registry.apcr:
        raise HTTPException(503, "APCR unavailable")

    t_start = time.perf_counter()
    try:
        metrics = compute_all_metrics(req.code)
        feat_vec = metrics_to_feature_vector(metrics)
    except Exception as e:
        raise HTTPException(400, f"Feature extraction failed: {e}")

    import numpy as np
    result = registry.apcr.predict(np.asarray(feat_vec, dtype=np.float32))
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 4)
    return out


@app.post("/analyze/dre", tags=["analysis"])
def analyze_dre(req: DRERequest):
    """
    Differential Risk Encoder (DRE) — predicts bug risk from feature
    DIFFERENCES between the current commit and the previous one.

    Novel: no JIT-SDP paper uses delta features as input representation.
    DRE encodes [delta, absolute, magnitude, direction] for each feature.

    Body:
        code:          Current commit source code
        prev_features: 15-dim static feature vector from the previous commit
                       (omit for first-commit or single-file analysis)

    Returns: risk_score, delta_contribution (how much delta features changed
    the score vs static-only), top_delta_features (most-changed features),
    static_score (absolute-only baseline), and model_type (mlp/logistic/heuristic).
    """
    if not registry.dre:
        raise HTTPException(503, "DRE unavailable")

    t_start = time.perf_counter()
    import numpy as np
    try:
        metrics = compute_all_metrics(req.code)
        x_curr = np.asarray(metrics_to_feature_vector(metrics), dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Feature extraction failed: {e}")

    x_prev = np.asarray(req.prev_features, dtype=np.float32) if req.prev_features else None
    result = registry.dre.predict(x_curr, x_prev)
    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 4)
    return out


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
# Batch analysis endpoint — analyse multiple files in one request
# ---------------------------------------------------------------------------

class BatchFileRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=500_000)
    filename: str = "snippet.py"
    language: Optional[str] = None
    git_metadata: Optional[dict] = None


class BatchAnalyzeRequest(BaseModel):
    files: list[BatchFileRequest] = Field(..., min_length=1, max_length=20)
    fail_fast: bool = False  # stop after first error


# ---------------------------------------------------------------------------
# Effort-aware triage endpoint
# ---------------------------------------------------------------------------

class TriageFileRequest(BaseModel):
    filename: str = Field(..., description="File path or name")
    code: str = Field(..., min_length=1, max_length=500_000, description="Source code content")
    loc: int = Field(0, description="Lines of code (estimated from code if 0)")


class TriageRequest(BaseModel):
    files: list[TriageFileRequest] = Field(..., min_length=1, max_length=50)
    top_k: int = Field(5, ge=1, le=20, description="How many top-risk files to highlight")


@app.post("/analyze/triage", tags=["analysis"])
def analyze_triage(req: TriageRequest):
    """
    Effort-aware triage ranking for a set of files.

    Runs lightweight security + bug prediction on each file and ranks them
    by a combined risk score weighted by lines of code (inspection effort).

    Returns the ranked file list, Precision@K, PofB@20 (what fraction of
    predicted-buggy files appear in the top 20%), and Effort@80 (what
    fraction of files you must inspect to cover 80% of the risk).

    Metrics follow Yang et al. 2016 (effort-aware JIT defect prediction)
    and Kamei et al. 2013 (JIT quality assurance).

    Body: { "files": [{"filename": ..., "code": ..., "loc": ...}], "top_k": 5 }
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))

    t_start = time.perf_counter()

    from evaluation.precision_at_k import (
        pofb_at_effort,
        effort_at_recall,
        precision_at_k as pak,
    )

    scored: list[dict] = []
    for f in req.files:
        loc = f.loc if f.loc > 0 else max(1, f.code.count("\n") + 1)
        language = detect_language(f.filename, None)

        # --- Bug probability (lightweight: metrics-based, no git) ---
        bug_prob = 0.0
        if registry.bug_predictor:
            try:
                bug_res = registry.bug_predictor.predict(f.code, None)
                bug_prob = float(bug_res.to_dict().get("bug_probability", 0.0))
            except Exception:
                pass

        # --- Security score ---
        sec_score = 0.0
        n_findings = 0
        if registry.security and language == "python" and not getattr(registry.security, "_ml_disabled", True):
            try:
                sec_score = registry.security.vulnerability_score(f.code)
                n_findings = len(registry.security.predict(f.code))
            except Exception:
                pass
        else:
            try:
                scan = scan_security_patterns_for_language(f.code, language)
                n_findings = scan.to_dict().get("summary", {}).get("total", 0)
                sec_score = min(0.95, n_findings * 0.15)
            except Exception:
                pass

        # Combined risk: security + bug weighted equally, then penalty for LOC
        # (larger files carry more absolute risk — effort-aware weighting)
        risk_score = 0.5 * sec_score + 0.5 * bug_prob
        # Effort-weighted: normalise by log(LOC) so a 1000-line file ranks
        # above a 10-line file with the same density (inspection prioritisation)
        effort_weighted_score = risk_score * (1.0 + 0.3 * (loc / 100.0) ** 0.5)

        scored.append({
            "filename": f.filename,
            "loc": loc,
            "bug_probability": round(bug_prob, 4),
            "vulnerability_score": round(sec_score, 4),
            "n_security_findings": n_findings,
            "risk_score": round(risk_score, 4),
            "effort_weighted_score": round(effort_weighted_score, 4),
            "language": language,
        })

    # Sort by effort-weighted score descending
    scored.sort(key=lambda x: x["effort_weighted_score"], reverse=True)

    # Assign triage rank
    for i, s in enumerate(scored):
        s["rank"] = i + 1
        s["triage_label"] = (
            "inspect_now" if i < req.top_k and s["risk_score"] >= 0.4
            else "review" if i < req.top_k
            else "low_priority"
        )

    # Effort-aware metrics over the ranked list
    import numpy as _np
    scores_arr = _np.array([s["effort_weighted_score"] for s in scored])
    bug_labels  = _np.array([1 if s["bug_probability"] >= 0.5 else 0 for s in scored])
    sec_labels  = _np.array([1 if s["vulnerability_score"] >= 0.4 else 0 for s in scored])
    combined_labels = _np.array([
        1 if (s["bug_probability"] >= 0.5 or s["vulnerability_score"] >= 0.4) else 0
        for s in scored
    ])

    k = min(req.top_k, len(scored))
    metrics: dict = {
        "n_files": len(scored),
        "top_k": k,
        "precision_at_k": round(pak(combined_labels, scores_arr, k), 4),
        "pofb_at_20pct": round(pofb_at_effort(bug_labels, scores_arr, effort=0.20), 4),
        "effort_at_80pct_recall": round(effort_at_recall(bug_labels, scores_arr, recall_target=0.80), 4),
        "n_high_risk": int((combined_labels == 1).sum()),
        "note": (
            "precision_at_k: fraction of top-K files that are genuinely risky. "
            "pofb_at_20pct: fraction of buggy files found when inspecting top 20%. "
            "effort_at_80pct_recall: how much of the list to inspect to find 80% of bugs."
        ),
    }

    return {
        "ranked_files": scored,
        "metrics": metrics,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
    }


@app.post("/analyze/batch", tags=["analysis"])
async def analyze_batch(req: BatchAnalyzeRequest, request: Request):
    """
    Analyse multiple files in one request. Files run in parallel (same
    ThreadPoolExecutor as /analyze). Returns per-file results plus an
    aggregate summary.

    - Max 20 files per request
    - Each file is independently cached
    - fail_fast=true returns 207 with partial results on first error
    """
    _check_rate_limit(request)
    loop = asyncio.get_running_loop()

    async def _analyse_one(f: BatchFileRequest) -> dict:
        cache_key = _cache_key(f.code, f.filename, f.git_metadata)
        _batch_cached = _cache_get(_analysis_cache, cache_key)
        if _batch_cached is not None:
            return {"filename": f.filename, "cached": True, **_batch_cached}

        # Reuse the full analysis logic via an internal AnalyzeRequest
        inner = AnalyzeRequest(
            code=f.code,
            filename=f.filename,
            language=f.language,
            git_metadata=f.git_metadata,
        )
        # Directly call the core analysis coroutine
        result = await analyze_full(inner, request)
        if hasattr(result, "model_dump"):
            return {"filename": f.filename, "cached": False, **result.model_dump()}
        return {"filename": f.filename, "cached": False, **result}

    results: list[dict] = []
    errors: list[dict] = []

    # Run all files concurrently
    tasks = [_analyse_one(f) for f in req.files]
    for coro in asyncio.as_completed(tasks):
        try:
            r = await coro
            results.append(r)
        except Exception as exc:
            errors.append({"error": str(exc)})
            if req.fail_fast:
                break

    # Aggregate stats
    scores = [r.get("overall_score", 0) for r in results if "overall_score" in r]
    total_sec = sum(r.get("security", {}).get("summary", {}).get("total", 0) for r in results)
    aggregate = {
        "files_analysed": len(results),
        "files_errored": len(errors),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "total_security_findings": total_sec,
        "critical_files": [
            r["filename"] for r in results
            if r.get("overall_score", 100) < 40 or r.get("status") == "critical"
        ],
    }

    payload = {
        "aggregate": aggregate,
        "results": results,
        "errors": errors,
    }
    # HTTP 207 Multi-Status when at least one file failed but others succeeded
    if errors:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=207, content=payload)
    return payload


@app.post("/analyze/batch/stream", tags=["analysis"])
async def analyze_batch_stream(req: BatchAnalyzeRequest, request: Request):
    """
    Streaming batch analysis — yields one Server-Sent Event per file as it completes,
    then a final 'done' event with the aggregate summary.

    Each SSE event has the shape:
        data: {"type": "file_result", "index": 0, "filename": "...", "result": {...}}
        data: {"type": "done", "aggregate": {...}}

    This lets the frontend render per-file progress bars rather than waiting for
    all files to finish before showing anything.
    """
    _check_rate_limit(request)
    loop = asyncio.get_running_loop()

    async def _generate():
        all_results: list[dict] = []
        errors: list[dict] = []

        async def _analyse_one(idx: int, f: BatchFileRequest) -> tuple[int, dict]:
            cache_key = _cache_key(f.code, f.filename, f.git_metadata)
            cached = _cache_get(_analysis_cache, cache_key)
            if cached is not None:
                return idx, {"filename": f.filename, "cached": True, **cached}
            inner = AnalyzeRequest(
                code=f.code,
                filename=f.filename,
                language=f.language,
                git_metadata=f.git_metadata,
            )
            result = await analyze_full(inner, request)
            out = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            return idx, {"filename": f.filename, "cached": False, **out}

        tasks = {asyncio.ensure_future(_analyse_one(i, f)): i for i, f in enumerate(req.files)}

        for coro in asyncio.as_completed(list(tasks.keys())):
            try:
                idx, r = await coro
                all_results.append(r)
                event = _json.dumps({"type": "file_result", "index": idx, "filename": r.get("filename"), "result": r})
                yield f"data: {event}\n\n"
            except Exception as exc:
                err = {"error": str(exc)}
                errors.append(err)
                yield f"data: {_json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
                if req.fail_fast:
                    break

        scores = [r.get("overall_score", 0) for r in all_results if "overall_score" in r]
        total_sec = sum(r.get("security", {}).get("summary", {}).get("total", 0) for r in all_results)
        aggregate = {
            "files_analysed": len(all_results),
            "files_errored": len(errors),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "total_security_findings": total_sec,
        }
        yield f"data: {_json.dumps({'type': 'done', 'aggregate': aggregate})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Explain endpoint — feature importance for ML predictions
# ---------------------------------------------------------------------------

@app.post("/analyze/explain", tags=["analysis"])
async def analyze_explain(req: AnalyzeRequest):
    """
    Returns feature-level explanations for the ML predictions on this code.

    For each ML model (complexity, security, bug):
    - Lists the top contributing features with their values and relative impact
    - Explains *why* the score is what it is (not just what the score is)

    Uses permutation-importance-style attribution from the trained models where
    available, falling back to threshold-based heuristics.
    """
    source = req.code
    loop = asyncio.get_running_loop()

    # Compute metrics (these are the model inputs)
    try:
        metrics = await loop.run_in_executor(_EXECUTOR, compute_all_metrics, source)
        from features.code_metrics import metrics_to_feature_vector
        feat_vec = metrics_to_feature_vector(metrics)
        feat_names = [
            "cyclomatic_complexity", "max_function_cc", "avg_function_cc",
            "sloc", "comments", "blank_lines",
            "halstead_volume", "halstead_difficulty", "halstead_effort",
            "bugs_delivered", "n_long_functions", "n_complex_functions",
            "max_line_length", "avg_line_length", "n_lines_over_80",
        ]
        features_dict = dict(zip(feat_names, [round(float(v), 3) for v in feat_vec]))
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    explanations: dict = {"features": features_dict, "models": {}}

    # ── Complexity explanation ──────────────────────────────────────────────
    try:
        if registry.complexity:
            model = registry.complexity._regressor  # XGBoost model
            import numpy as np
            x = np.array(feat_vec, dtype=float).reshape(1, -1)
            pred_cog = float(model.predict(x)[0])

            # Get feature importances from the trained model
            importances = model.feature_importances_  # shape (15,)
            ranked = sorted(
                zip(feat_names, feat_vec, importances),
                key=lambda t: t[2], reverse=True
            )
            top = [
                {
                    "feature": name,
                    "value": round(float(val), 2),
                    "importance": round(float(imp), 4),
                    "impact": "high" if imp > 0.15 else "medium" if imp > 0.05 else "low",
                }
                for name, val, imp in ranked[:8]
            ]
            explanations["models"]["complexity"] = {
                "predicted_cognitive_complexity": round(pred_cog, 1),
                "score": registry.complexity.predict(source).score,
                "top_features": top,
                "interpretation": (
                    f"The primary driver of complexity is '{ranked[0][0]}' "
                    f"(value={round(float(ranked[0][1]), 1)}, importance={round(float(ranked[0][2]), 3)})."
                ),
            }
    except Exception as e:
        explanations["models"]["complexity"] = {"error": str(e)}

    # ── Security explanation ────────────────────────────────────────────────
    try:
        from features.security_patterns import scan_security_patterns
        sec_scan = scan_security_patterns(source)
        vuln_categories: dict[str, int] = {}
        for f in sec_scan.findings:
            cat = getattr(f, "vuln_type", getattr(f, "category", "unknown"))
            vuln_categories[cat] = vuln_categories.get(cat, 0) + 1

        rf_feat_names = [
            "n_tokens", "n_defs", "n_imports", "n_calls", "n_attrs",
            "n_lines", "n_try", "n_except", "n_return", "n_raise",
            "n_os_calls", "n_subprocess", "n_eval", "n_exec", "n_open", "vocab_size",
        ]
        try:
            from models.security_detection import _build_rf_feature_vector
            rf_feats = _build_rf_feature_vector(source)
            rf_dict = dict(zip(rf_feat_names, [round(float(v), 2) for v in rf_feats[:16]]))
        except Exception:
            rf_dict = {}

        explanations["models"]["security"] = {
            "pattern_findings": len(sec_scan.findings),
            "finding_categories": vuln_categories,
            "rf_features": rf_dict,
            "high_risk_signals": [
                name for name, val in rf_dict.items()
                if name in ("n_eval", "n_exec", "n_subprocess", "n_os_calls") and val > 0
            ],
            "interpretation": (
                f"{len(sec_scan.findings)} pattern-based finding(s) detected. "
                + (f"High-risk calls: {[n for n in ('n_eval','n_exec','n_subprocess') if rf_dict.get(n,0) > 0]}."
                   if rf_dict else "")
            ),
        }
    except Exception as e:
        explanations["models"]["security"] = {"error": str(e)}

    # ── Bug risk explanation ────────────────────────────────────────────────
    try:
        if registry.bug_predictor:
            import numpy as np
            static_feats = np.array(feat_vec, dtype=float)
            # Feature→risk mapping (domain knowledge)
            bug_signals = {
                "cyclomatic_complexity": (float(feat_vec[0]), 10.0, "high CC correlates with defects"),
                "n_complex_functions":   (float(feat_vec[11]), 1.0,  "complex functions are bug-prone"),
                "n_long_functions":      (float(feat_vec[10]), 1.0,  "long functions harder to test"),
                "halstead_effort":       (float(feat_vec[8]),  5000, "high effort = more error-prone"),
                "bugs_delivered":        (float(feat_vec[9]),  0.5,  "Halstead bug estimate"),
            }
            drivers = [
                {
                    "feature": name,
                    "value": round(val, 2),
                    "threshold": thresh,
                    "exceeded": val > thresh,
                    "explanation": desc,
                }
                for name, (val, thresh, desc) in bug_signals.items()
            ]
            pred = registry.bug_predictor.predict(source)
            explanations["models"]["bug"] = {
                "bug_probability": pred.bug_probability,
                "risk_level": pred.risk_level,
                "risk_drivers": drivers,
                "risk_factors": pred.risk_factors,
                "interpretation": (
                    f"Bug probability {pred.bug_probability:.2f} ({pred.risk_level} risk). "
                    f"Main driver: {max(bug_signals.items(), key=lambda t: t[1][0]/max(t[1][1],0.001))[0]}."
                ),
            }
    except Exception as e:
        explanations["models"]["bug"] = {"error": str(e)}

    # ── Actionable summary ──────────────────────────────────────────────────
    actions = []
    cc = float(feat_vec[0]) if feat_vec else 0
    if cc > 15:
        actions.append(f"Reduce cyclomatic complexity from {cc:.0f} (target: <10) by extracting helper functions.")
    if float(feat_vec[10] if feat_vec else 0) > 0:
        actions.append(f"Break down {int(feat_vec[10])} long function(s) (>50 lines) into smaller units.")
    if float(feat_vec[11] if feat_vec else 0) > 0:
        actions.append(f"Refactor {int(feat_vec[11])} complex function(s) (CC>10).")
    if float(feat_vec[14] if feat_vec else 0) > 5:
        actions.append(f"Shorten {int(feat_vec[14])} line(s) exceeding 80 chars for readability.")
    if not actions:
        actions.append("No major structural issues detected.")

    explanations["actionable_summary"] = actions
    return explanations


# ---------------------------------------------------------------------------
# Feedback endpoint — stores thumbs-up/down per analysis finding
# ---------------------------------------------------------------------------

_feedback_store: list[dict] = []  # seeded from disk on first feedback_stats call

class FeedbackRequest(BaseModel):
    analysis_id: Optional[str] = None
    finding_key: Optional[str] = Field(default="analysis_overall", description="Unique key for the finding or 'analysis_overall'")
    verdict: Literal["helpful", "not_helpful", "false_positive"] = Field(...)
    comment: Optional[str] = Field(default=None, max_length=500)

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
    """
    Record user feedback on a specific analysis finding.

    Persists to feedback.jsonl. When the feedback file accumulates >= 50
    corrected samples for a given task, triggers background active learning
    fine-tuning (non-blocking: runs in the background thread pool).
    """
    entry = {
        "analysis_id": req.analysis_id,
        "finding_key": req.finding_key,
        "verdict": req.verdict,
        "comment": req.comment,
        "ts": time.time(),
        # Map verdict to active learning label
        "task": _infer_task_from_finding(req.finding_key),
        "user_label": _verdict_to_label(req.verdict),
    }
    _feedback_store.append(entry)
    # Persist to disk so feedback survives restarts
    try:
        with open(_FEEDBACK_PATH, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Could not persist feedback to disk: %s", exc)

    # Trigger active learning when enough samples have accumulated
    _maybe_trigger_active_learning(entry.get("task", ""))

    return {"status": "ok", "stored": len(_feedback_store)}


def _infer_task_from_finding(finding_key: str) -> str:
    """Infer the ML task from a finding key string."""
    if not finding_key:
        return "unknown"
    key = finding_key.lower()
    if any(k in key for k in ("vuln", "security", "inject", "xss", "sqli", "crypto")):
        return "security"
    if any(k in key for k in ("bug", "defect", "risk")):
        return "bug"
    if any(k in key for k in ("complex", "cc", "cognitive")):
        return "complexity"
    return "unknown"


def _verdict_to_label(verdict: str) -> int:
    """Convert a user verdict string to a binary active-learning label."""
    v = (verdict or "").lower().strip()
    # "false_positive" means model said vulnerable/buggy but code is clean
    if v in ("false_positive", "not_helpful", "wrong", "incorrect"):
        return 0   # model over-predicted -- correct label is negative
    # "helpful", "correct" means model was right -- reinforce positive label
    if v in ("helpful", "correct", "true_positive"):
        return 1
    return -1   # unknown -- skip in active learning


_AL_TRIGGER_THRESHOLD = int(os.environ.get("AL_TRIGGER_THRESHOLD", "50"))
_al_feedback_counts: dict[str, int] = {}
# History of active learning rounds for the /feedback/al_status endpoint.
# Each entry: {task, triggered_at, pre_auc, post_auc, pre_ece, post_ece, deployed, reason}
_al_history: list[dict] = []


def _maybe_trigger_active_learning(task: str) -> None:
    """
    Check if enough feedback has accumulated to trigger active learning.
    Runs the fine-tuner in the background thread pool (non-blocking).
    """
    if not task or task == "unknown":
        return

    _al_feedback_counts[task] = _al_feedback_counts.get(task, 0) + 1
    count = _al_feedback_counts[task]

    if count < _AL_TRIGGER_THRESHOLD:
        return

    # Reset counter so we don't re-trigger immediately
    _al_feedback_counts[task] = 0
    logger.info(
        "Active learning triggered for task=%s (%d feedback samples reached)",
        task, count,
    )

    def _run_al():
        import datetime as _dt
        entry: dict = {
            "task": task,
            "triggered_at": _dt.datetime.utcnow().isoformat() + "Z",
            "deployed": False,
            "reason": None,
            "pre_auc": None,
            "post_auc": None,
            "pre_ece": None,
            "post_ece": None,
        }
        try:
            from training.active_learning import ActiveLearner
            learner = ActiveLearner(task=task, min_feedback_samples=_AL_TRIGGER_THRESHOLD)
            result = learner.run()
            entry["deployed"] = bool(result.deployed)
            entry["reason"] = getattr(result, "reason", None)
            entry["pre_ece"] = getattr(result, "pre_ece", None)
            entry["post_ece"] = getattr(result, "post_ece", None)
            entry["pre_auc"] = getattr(result, "pre_auc", None)
            entry["post_auc"] = getattr(result, "temporal_auc", None)
            if result.deployed:
                logger.info(
                    "Active learning deployed for task=%s: ECE %.4f -> %.4f  AUC=%.3f",
                    task, result.pre_ece, result.post_ece, result.temporal_auc,
                )
            else:
                logger.info(
                    "Active learning skipped for task=%s: %s", task, result.reason
                )
        except Exception as exc:
            logger.warning("Active learning run failed for task=%s: %s", task, exc)
            entry["reason"] = str(exc)
        finally:
            _al_history.append(entry)

        # After a successful security AL round, run hard negative mining to
        # tighten the model on false positives (completes the feedback loop).
        if entry.get("deployed") and task == "security":
            try:
                from training.hard_negative_miner import mine_hard_negatives
                dataset = "data/security_dataset.jsonl"
                hn_output = "data/security_dataset_hn.jsonl"
                if Path(dataset).exists():
                    n_hn = mine_hard_negatives(
                        clean_dirs=["data/clean_repos"] if Path("data/clean_repos").exists() else [],
                        dataset_path=dataset,
                        output_path=hn_output,
                        threshold=0.5,
                        max_hn=200,
                    )
                    if n_hn > 0:
                        logger.info(
                            "Hard negative mining: added %d false-positive samples to %s",
                            n_hn, hn_output,
                        )
                        entry["hard_negatives_mined"] = n_hn
            except Exception as hn_exc:
                logger.debug("Hard negative mining skipped: %s", hn_exc)

    # Submit to background executor (non-blocking, won't delay the /feedback response)
    try:
        _EXECUTOR.submit(_run_al)
    except Exception as exc:
        logger.warning("Could not submit AL job to executor: %s", exc)

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


@app.get("/feedback/al_status", tags=["meta"])
def al_status():
    """
    Active learning loop status.

    Returns per-task feedback progress toward next retrain trigger, the
    configured trigger threshold, and a history of past retrain rounds
    (deployed/skipped, AUC before/after, ECE before/after).
    """
    per_task: dict[str, dict] = {}
    all_tasks = {"security", "bug", "complexity", "pattern"}
    for t in all_tasks:
        count = _al_feedback_counts.get(t, 0)
        per_task[t] = {
            "feedback_since_last_trigger": count,
            "needed_for_next_trigger": max(0, _AL_TRIGGER_THRESHOLD - count),
            "progress_pct": round(min(100, count / _AL_TRIGGER_THRESHOLD * 100), 1),
        }

    return {
        "trigger_threshold": _AL_TRIGGER_THRESHOLD,
        "per_task": per_task,
        "total_feedback": len(_feedback_store),
        "history": list(reversed(_al_history)),  # most recent first
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

    # Pass token via URL fragment — fragments are never sent to servers or logged by proxies.
    # The frontend reads window.location.hash and clears it immediately.
    return RedirectResponse(f"{frontend_url}/#github_token={access_token}")


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
# GitHub PR analysis + comment posting
# ---------------------------------------------------------------------------

class PRAnalyzeRequest(BaseModel):
    pr_url: str = Field(..., description="Full GitHub PR URL e.g. https://github.com/owner/repo/pull/123")
    github_token: str = Field(..., description="GitHub personal access token or OAuth token")
    post_comments: bool = Field(True, description="Whether to post review comments to GitHub")
    max_files: int = Field(10, ge=1, le=20, description="Max files to analyze")


def _gh_api(path: str, token: str, method: str = "GET", body: dict | None = None) -> dict:
    """Make a GitHub API call. Raises on HTTP error."""
    url = f"https://api.github.com{path}"
    data = _json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return _json.loads(resp.read())
    except urllib.request.HTTPError as e:
        body_text = e.read().decode(errors="replace")[:300]
        raise HTTPException(status_code=e.code, detail=f"GitHub API error: {body_text}")


def _parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse https://github.com/owner/repo/pull/123 → (owner, repo, pr_number)."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url.strip())
    if not m:
        raise HTTPException(400, "Invalid PR URL. Expected: https://github.com/owner/repo/pull/123")
    return m.group(1), m.group(2), int(m.group(3))


def _fetch_pr_file_content(owner: str, repo: str, path: str, ref: str, token: str) -> str:
    """Fetch raw file content at a specific git ref."""
    try:
        data = _gh_api(f"/repos/{owner}/{repo}/contents/{path}?ref={ref}", token)
        import base64
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _generate_fix_suggestion(vuln_type: str, line: str, description: str, snippet: str = "") -> str:
    """
    Generate a before/after fix block for a security finding.
    Uses the actual code line when available, falls back to the finding snippet,
    then to a canonical example. Always returns a non-empty markdown block.
    """
    # Pick the best available "before" code
    before = (line or snippet or "").rstrip()

    # ── Canonical fix examples used when we have no real code ──
    EXAMPLES: dict[str, tuple[str, str]] = {
        "sql_injection": (
            'query = "SELECT * FROM users WHERE id = " + user_id',
            'query = "SELECT * FROM users WHERE id = %s"\ncursor.execute(query, (user_id,))',
        ),
        "command_injection": (
            'os.system("ls " + user_input)',
            'subprocess.run(["ls", user_input], check=True)  # shell=False by default',
        ),
        "hardcoded_secret": (
            'api_key = "sk-prod-abc123xyz"',
            'import os\napi_key = os.environ["API_KEY"]  # store secrets in env vars',
        ),
        "hardcoded_password": (
            'password = "hunter2"',
            'import os\npassword = os.environ["DB_PASSWORD"]',
        ),
        "xss": (
            'element.innerHTML = user_input',
            'element.textContent = user_input  # or use DOMPurify.sanitize()',
        ),
        "path_traversal": (
            'open(base_dir + "/" + user_filename)',
            'safe = os.path.realpath(os.path.join(base_dir, user_filename))\n'
            'if not safe.startswith(os.path.realpath(base_dir)):\n'
            '    raise ValueError("Path traversal detected")\n'
            'open(safe)',
        ),
        "eval_injection": (
            'result = eval(user_input)',
            'import ast\nresult = ast.literal_eval(user_input)  # safe eval for literals only',
        ),
        "exec_injection": (
            'exec(user_code)',
            '# Remove exec() — run user code in a sandboxed subprocess instead',
        ),
        "open_redirect": (
            'return redirect(request.args.get("next"))',
            'from urllib.parse import urlparse\nnext_url = request.args.get("next", "/")\n'
            'if urlparse(next_url).netloc:  # block absolute URLs\n'
            '    next_url = "/"\nreturn redirect(next_url)',
        ),
        "ssrf": (
            'requests.get(user_url)',
            '# Validate URL against an allowlist before fetching\n'
            'ALLOWED_HOSTS = {"api.example.com"}\n'
            'parsed = urllib.parse.urlparse(user_url)\n'
            'if parsed.hostname not in ALLOWED_HOSTS:\n'
            '    raise ValueError("Blocked host")\n'
            'requests.get(user_url, timeout=5)',
        ),
        "insecure_deserialization": (
            'data = pickle.loads(user_bytes)',
            '# Never deserialize untrusted data with pickle\n'
            '# Use json.loads() or a schema-validated format instead\n'
            'data = json.loads(user_bytes)',
        ),
        "weak_crypto": (
            'hashlib.md5(password.encode()).hexdigest()',
            'import bcrypt\nhashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())',
        ),
        "debug_enabled": (
            'app.run(debug=True)',
            'app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")',
        ),
        "insecure_random": (
            'token = random.randint(0, 999999)',
            'import secrets\ntoken = secrets.token_hex(32)  # cryptographically secure',
        ),
        "missing_auth": (
            '@app.route("/admin")\ndef admin():',
            '@app.route("/admin")\n@login_required  # add authentication\ndef admin():',
        ),
    }

    def _make_block(bad: str, good: str) -> str:
        return (
            f"**Before (vulnerable):**\n```python\n{bad}\n```\n\n"
            f"**After (fixed):**\n```python\n{good}\n```"
        )

    # Try to auto-fix the actual line
    if before:
        stripped = before

        if vuln_type == "sql_injection":
            m = re.search(r'(["\'][^"\']*["\'])\s*\+\s*(\w+)', stripped)
            if m:
                var = m.group(2)
                fixed = re.sub(r'["\'][^"\']*["\'][\s]*\+[\s]*\w+',
                               f'"%s"  # then: cursor.execute(query, ({var},))', stripped, count=1)
                return _make_block(stripped, fixed)

        if vuln_type == "command_injection" and "os.system" in stripped:
            fixed = re.sub(r'os\.system\((.+)\)',
                           r'subprocess.run(\1, shell=False)  # pass args as list', stripped)
            if fixed != stripped:
                return _make_block(stripped, fixed)

        if vuln_type in ("hardcoded_secret", "hardcoded_password"):
            m = re.match(r'^(\s*)([\w.]+)\s*=\s*["\'].*["\']', stripped)
            if m:
                indent, var = m.group(1), m.group(2)
                env_key = re.sub(r'[^A-Z0-9]', '_', var.upper())
                fixed = f'{indent}{var} = os.environ["{env_key}"]'
                return _make_block(stripped, fixed)

        if vuln_type == "xss":
            fixed = stripped.replace("innerHTML", "textContent")
            if fixed != stripped:
                return _make_block(stripped, fixed)

        if vuln_type in ("eval_injection",):
            fixed = stripped.replace("eval(", "ast.literal_eval(")
            if fixed != stripped:
                return _make_block(stripped, fixed)

        if vuln_type in ("insecure_random",) and "random." in stripped:
            fixed = re.sub(r'random\.\w+\([^)]*\)', 'secrets.token_hex(32)', stripped)
            if fixed != stripped:
                return _make_block(stripped, fixed)

        if vuln_type == "debug_enabled":
            fixed = stripped.replace("debug=True", 'debug=os.environ.get("DEBUG","false")=="true"')
            if fixed != stripped:
                return _make_block(stripped, fixed)

    # Fall back to canonical example for this vuln type
    if vuln_type in EXAMPLES:
        bad, good = EXAMPLES[vuln_type]
        return _make_block(bad, good)

    # Last resort — generic advice
    return (
        f"**Issue:** {description}\n\n"
        f"No automated fix template available for `{vuln_type}`.\n\n"
        f"**General guidance:** validate and sanitize all untrusted input, "
        f"apply the principle of least privilege, and consult OWASP for `{vuln_type}` remediation."
    )


@app.post("/github/analyze-pr", tags=["github"])
async def analyze_pr(req: PRAnalyzeRequest):
    """
    Fetch a GitHub PR's changed files, run analysis, and optionally post
    review comments inline on risky lines + a summary comment.

    Returns the full analysis results per file plus GitHub comment URLs.
    """
    owner, repo, pr_number = _parse_pr_url(req.pr_url)
    token = req.github_token

    loop = asyncio.get_running_loop()

    # 1. Fetch PR metadata + file list
    def _fetch_pr():
        pr   = _gh_api(f"/repos/{owner}/{repo}/pulls/{pr_number}", token)
        files = _gh_api(f"/repos/{owner}/{repo}/pulls/{pr_number}/files", token)
        return pr, files

    pr_data, files_data = await loop.run_in_executor(_EXECUTOR, _fetch_pr)

    head_sha = pr_data.get("head", {}).get("sha", "HEAD")
    pr_title = pr_data.get("title", "")

    # 2. Filter to analyzable files
    EXT_RE = re.compile(r"\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|cs|rb|php|kt|swift)$")
    changed_files = [
        f for f in files_data
        if EXT_RE.search(f.get("filename", "")) and f.get("status") != "removed"
    ][:req.max_files]

    if not changed_files:
        return {"status": "no_analyzable_files", "pr_title": pr_title, "files": []}

    # 3. Fetch + analyze each file in parallel
    async def _analyze_file(file_info: dict) -> dict:
        filename = file_info["filename"]
        language = detect_language(filename, "python")
        source = await loop.run_in_executor(
            _EXECUTOR,
            lambda: _fetch_pr_file_content(owner, repo, filename, head_sha, token)
        )
        if not source or len(source) < 10:
            return {"filename": filename, "skipped": True}

        # Truncate large files
        lines = source.splitlines()
        if len(lines) > _MAX_SLOC:
            source = "\n".join(lines[:_MAX_SLOC])

        def _run_analysis():
            sec   = _step_security(source, language)
            cpx, _= _step_complexity(source)
            bug, _= _step_bug(source, None)
            fr    = _step_function_risk(source)
            return sec, cpx, bug, fr

        sec_out, cpx_out, bug_out, fr_out = await loop.run_in_executor(_EXECUTOR, _run_analysis)

        return {
            "filename": filename,
            "patch": file_info.get("patch", "")[:500],
            "security": {
                "findings": sec_out.get("findings", []),
                "summary":  sec_out.get("summary", {}),
            },
            "complexity": cpx_out.model_dump() if cpx_out else {},
            "bug_prediction": bug_out.model_dump() if bug_out else {},
            "function_risk": fr_out,
            "_source": source,   # kept server-side for suggestion generation, stripped from response
            "skipped": False,
        }

    file_results = await asyncio.gather(*[_analyze_file(f) for f in changed_files])

    # 4. Post GitHub review comments if requested
    comment_urls = []
    if req.post_comments:
        # Build review body (summary comment)
        n_sec = sum(
            len(r.get("security", {}).get("findings", []))
            for r in file_results if not r.get("skipped")
        )
        high_bug = [
            r["filename"] for r in file_results
            if not r.get("skipped") and r.get("bug_prediction", {}).get("risk_level") in ("high", "critical")
        ]

        summary_body = (
            f"## IntelliCode Review\n\n"
            f"Analyzed **{len([r for r in file_results if not r.get('skipped')])}** file(s) "
            f"in PR #{pr_number}: _{pr_title}_\n\n"
        )
        if n_sec:
            summary_body += f"- **{n_sec} security finding(s)** detected\n"
        if high_bug:
            summary_body += f"- High bug risk: {', '.join(f'`{f}`' for f in high_bug)}\n"

        # Add per-file function risk summary
        for r in file_results:
            if r.get("skipped"):
                continue
            fr = r.get("function_risk") or {}
            top_k = fr.get("top_k", [])
            if top_k:
                summary_body += f"\n**`{r['filename']}`** — top risk functions:\n"
                for fn in top_k[:3]:
                    summary_body += (
                        f"- `{fn['name']}()` L{fn['lineno']}–{fn['end_lineno']} "
                        f"({fn['risk_level']}) — {fn['reason']}\n"
                    )

        summary_body += "\n---\n_Posted by [IntelliCode](https://github.com/safaapatel/intellcode)_"

        def _post_summary():
            return _gh_api(
                f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
                token, method="POST",
                body={"body": summary_body},
            )

        try:
            comment = await loop.run_in_executor(_EXECUTOR, _post_summary)
            comment_urls.append(comment.get("html_url", ""))
        except Exception as e:
            logger.warning("Failed to post PR summary comment: %s", e)

        # Post inline review comments for security findings with suggestions
        inline_comments = []
        for r in file_results:
            if r.get("skipped"):
                continue
            # Build a lookup of line → code text for suggestion blocks
            file_source = r.get("_source", "")
            source_lines = file_source.splitlines() if file_source else []

            for finding in r.get("security", {}).get("findings", []):
                lineno = finding.get("lineno", 1)
                if not lineno or lineno <= 0:
                    continue
                vuln_type  = finding.get("vuln_type", "")
                title      = finding.get("title", "")
                severity   = finding.get("severity", "low").upper()
                description = finding.get("description", "")
                cwe        = finding.get("cwe", "")
                confidence = finding.get("confidence", 0)

                # Try to get the actual offending line for the suggestion
                code_line = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
                fix_block = _generate_fix_suggestion(vuln_type, code_line, description)

                body = (
                    f"### {severity} — {title}\n\n"
                    f"{description}\n\n"
                    f"**CWE:** `{cwe}` &nbsp;|&nbsp; **Confidence:** {confidence:.0%}\n\n"
                    f"#### Suggested fix\n\n"
                    f"{fix_block}"
                )

                inline_comments.append({
                    "path": r["filename"],
                    "line": lineno,
                    "body": body,
                })

        if inline_comments:
            def _post_review():
                return _gh_api(
                    f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
                    token, method="POST",
                    body={
                        "commit_id": head_sha,
                        "body": "IntelliCode security findings (inline)",
                        "event": "COMMENT",
                        "comments": inline_comments[:10],
                    },
                )
            try:
                review = await loop.run_in_executor(_EXECUTOR, _post_review)
                comment_urls.append(review.get("html_url", ""))
            except Exception as e:
                logger.warning("Inline review failed (%s) — falling back to single comment", e)
                # Fallback: post all findings as one regular PR comment
                fallback_body = "## IntelliCode Security Findings & Suggested Fixes\n"
                for ic in inline_comments[:10]:
                    fallback_body += f"\n---\n\n**`{ic['path']}`** L{ic['line']}\n\n{ic['body']}\n"
                fallback_body += "\n---\n_Posted by [IntelliCode Review](https://intellcode.onrender.com)_"
                try:
                    fb = _gh_api(
                        f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
                        token, method="POST", body={"body": fallback_body},
                    )
                    comment_urls.append(fb.get("html_url", ""))
                except Exception as e2:
                    logger.warning("Fallback comment also failed: %s", e2)

    # Strip internal _source field before returning
    clean_results = [{k: v for k, v in r.items() if k != "_source"} for r in file_results]

    return {
        "status": "ok",
        "pr_url": req.pr_url,
        "pr_title": pr_title,
        "files_analyzed": len([r for r in clean_results if not r.get("skipped")]),
        "comment_urls": comment_urls,
        "results": clean_results,
    }


# ---------------------------------------------------------------------------
# POST /github/create-pr  — commit code + open a PR with analysis findings
# ---------------------------------------------------------------------------

class CreatePRRequest(BaseModel):
    github_token: str
    repo_full_name: str          # "owner/repo"
    filename: str                # path inside the repo, e.g. "src/utils.py"
    code: str
    branch_name: str = ""        # auto-generated if empty
    pr_title: str = ""
    base_branch: str = ""        # defaults to repo default branch
    analysis_summary: str = ""   # pre-built markdown summary
    security_findings: list = [] # list of {vuln_type, title, severity, description, cwe, confidence, lineno}


def _b64encode(s: str) -> str:
    import base64
    return base64.b64encode(s.encode()).decode()


@app.post("/github/create-pr", tags=["github"])
async def create_pr(req: CreatePRRequest):
    """
    1. Create a new branch off the repo's default branch
    2. Commit `code` to `filename` on that branch
    3. Open a pull request
    4. Post `analysis_summary` as a PR comment
    Returns { pr_url, pr_number, comment_url, branch }
    """
    token = req.github_token
    owner, repo = req.repo_full_name.split("/", 1)
    loop = asyncio.get_running_loop()

    def _setup():
        # Get repo default branch if not supplied
        repo_info = _gh_api(f"/repos/{owner}/{repo}", token)
        base = req.base_branch or repo_info.get("default_branch", "main")

        # Get SHA of tip of base branch
        ref_data = _gh_api(f"/repos/{owner}/{repo}/git/ref/heads/{base}", token)
        base_sha = ref_data["object"]["sha"]
        return base, base_sha

    base_branch, base_sha = await loop.run_in_executor(_EXECUTOR, _setup)

    # Generate branch name if not provided
    import time as _time
    ts = int(_time.time())
    branch = req.branch_name.strip() or f"intellcode/review-{ts}"
    # Sanitize branch name
    branch = re.sub(r"[^a-zA-Z0-9._/-]", "-", branch)

    def _create_branch():
        return _gh_api(
            f"/repos/{owner}/{repo}/git/refs",
            token, method="POST",
            body={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )

    try:
        await loop.run_in_executor(_EXECUTOR, _create_branch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not create branch: {e}")

    # Check if file already exists (need its SHA to update)
    def _get_existing():
        try:
            existing = _gh_api(f"/repos/{owner}/{repo}/contents/{req.filename}?ref={branch}", token)
            return existing.get("sha")
        except Exception:
            return None

    existing_sha = await loop.run_in_executor(_EXECUTOR, _get_existing)

    def _commit_file():
        body: dict = {
            "message": f"Add {req.filename} via IntelliCode Review",
            "content": _b64encode(req.code),
            "branch": branch,
        }
        if existing_sha:
            body["sha"] = existing_sha
        return _gh_api(
            f"/repos/{owner}/{repo}/contents/{req.filename}",
            token, method="PUT", body=body,
        )

    await loop.run_in_executor(_EXECUTOR, _commit_file)

    # Create the PR
    pr_title = req.pr_title.strip() or f"IntelliCode Review: {req.filename}"

    def _open_pr():
        return _gh_api(
            f"/repos/{owner}/{repo}/pulls",
            token, method="POST",
            body={
                "title": pr_title,
                "head": branch,
                "base": base_branch,
                "body": (
                    f"## IntelliCode Automated Review\n\n"
                    f"This PR was created automatically by [IntelliCode](https://intellcode.onrender.com) "
                    f"after analysing `{req.filename}`.\n\n"
                    f"{req.analysis_summary or '_No analysis summary provided._'}"
                ),
            },
        )

    pr_data = await loop.run_in_executor(_EXECUTOR, _open_pr)
    pr_url    = pr_data.get("html_url", "")
    pr_number = pr_data.get("number")
    head_sha  = pr_data.get("head", {}).get("sha") or base_sha

    comment_url = ""

    # Build a single PR comment with summary + per-finding suggested fixes.
    # We use a regular issue comment (not inline review) so it always renders
    # regardless of whether the file diff is empty.
    source_lines = req.code.splitlines() if req.code else []

    findings_md = ""
    for finding in req.security_findings[:10]:
        lineno     = finding.get("lineno", 0)
        code_line  = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
        vuln_type  = finding.get("vuln_type", "")
        title      = finding.get("title", "")
        severity   = finding.get("severity", "low").upper()
        description = finding.get("description", "")
        cwe        = finding.get("cwe", "")
        confidence = finding.get("confidence", 0)
        snippet    = finding.get("snippet", "")
        fix_block  = _generate_fix_suggestion(vuln_type, code_line, description, snippet)

        findings_md += (
            f"\n---\n\n"
            f"#### {severity} — {title} (L{lineno})\n\n"
            f"{description}\n\n"
            f"**CWE:** `{cwe}` &nbsp;|&nbsp; **Confidence:** {confidence:.0%}\n\n"
            f"**Suggested fix:**\n\n{fix_block}\n"
        )

    full_body = req.analysis_summary or ""
    if findings_md:
        full_body += f"\n\n## Security Findings & Suggested Fixes\n{findings_md}"
    full_body += "\n\n---\n_Posted by [IntelliCode Review](https://intellcode.onrender.com)_"

    if full_body and pr_number:
        def _post_comment():
            return _gh_api(
                f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
                token, method="POST",
                body={"body": full_body},
            )
        try:
            comment = await loop.run_in_executor(_EXECUTOR, _post_comment)
            comment_url = comment.get("html_url", "")
        except Exception as e:
            logger.warning("Could not post PR comment: %s", e)

    # Create GitHub Issues for critical findings so they're tracked permanently
    issue_urls: list[str] = []
    critical_findings = [
        f for f in req.security_findings
        if f.get("severity") in ("critical",)
    ]
    for finding in critical_findings[:5]:  # cap at 5 issues
        lineno      = finding.get("lineno", 0)
        code_line   = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
        vuln_type   = finding.get("vuln_type", "")
        title       = finding.get("title", "")
        description = finding.get("description", "")
        cwe         = finding.get("cwe", "")
        confidence  = finding.get("confidence", 0)
        snippet     = finding.get("snippet", "")
        fix_block   = _generate_fix_suggestion(vuln_type, code_line, description, snippet)

        issue_body = (
            f"## Critical Security Finding\n\n"
            f"**File:** `{req.filename}` — Line {lineno}\n"
            f"**CWE:** `{cwe}` &nbsp;|&nbsp; **Confidence:** {confidence:.0%}\n\n"
            f"### Description\n\n{description}\n\n"
            f"### Suggested Fix\n\n{fix_block}\n\n"
            f"### Context\n\n"
            f"Detected in PR #{pr_number}: [{pr_title}]({pr_url})\n\n"
            f"---\n_Opened automatically by [IntelliCode Review](https://intellcode.onrender.com)_"
        )

        def _make_issue(t=title, b=issue_body):
            return _gh_api(
                f"/repos/{owner}/{repo}/issues",
                token, method="POST",
                body={
                    "title": f"[Security] {t} in {req.filename}",
                    "body": b,
                    "labels": ["security", "intellcode", "critical"],
                },
            )
        try:
            issue = await loop.run_in_executor(_EXECUTOR, _make_issue)
            issue_urls.append(issue.get("html_url", ""))
        except Exception as e:
            logger.warning("Could not create issue for finding '%s': %s", title, e)

    return {
        "pr_url": pr_url,
        "pr_number": pr_number,
        "branch": branch,
        "comment_url": comment_url,
        "issue_urls": issue_urls,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
