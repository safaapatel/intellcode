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
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features.code_metrics import compute_all_metrics
from features.ast_extractor import extract_ast_features
from features.security_patterns import scan_security_patterns
from models.security_detection import EnsembleSecurityModel
from models.complexity_prediction import ComplexityPredictionModel
from models.bug_predictor import BugPredictionModel, GitMetadata


# ---------------------------------------------------------------------------
# Model singleton registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    security: EnsembleSecurityModel | None = None
    complexity: ComplexityPredictionModel | None = None
    bug_predictor: BugPredictionModel | None = None
    pattern_model = None   # loaded lazily (large download)
    load_errors: dict[str, str] = {}


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Loading ML models...")

    try:
        registry.security = EnsembleSecurityModel(
            checkpoint_dir="checkpoints/security"
        )
        print("  [OK] Security detection model")
    except Exception as e:
        registry.load_errors["security"] = str(e)
        print(f"  [WARN] Security model: {e}")

    try:
        registry.complexity = ComplexityPredictionModel(
            checkpoint_path="checkpoints/complexity/model.pkl"
        )
        print("  [OK] Complexity prediction model")
    except Exception as e:
        registry.load_errors["complexity"] = str(e)
        print(f"  [WARN] Complexity model: {e}")

    try:
        registry.bug_predictor = BugPredictionModel(
            checkpoint_dir="checkpoints/bug_predictor"
        )
        print("  [OK] Bug prediction model")
    except Exception as e:
        registry.load_errors["bug_predictor"] = str(e)
        print(f"  [WARN] Bug predictor: {e}")

    print("Models ready.")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IntelliCode Review API",
    description="AI-powered code analysis REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Python source code to analyze", min_length=1)
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
    overall_score: int
    status: str            # "clean" | "action_required" | "critical"
    summary: str


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
            "pattern_recognition": "not_loaded",  # lazy
        },
        "errors": registry.load_errors,
    }


@app.get("/models", tags=["meta"])
def list_models():
    """Returns metadata about each available ML model."""
    return {
        "models": [
            {
                "id": "pattern_recognition",
                "name": "Pattern Recognition Model",
                "architecture": "Fine-tuned CodeBERT (microsoft/codebert-base)",
                "status": "production",
                "accuracy": 0.87,
                "tech_stack": ["PyTorch", "HuggingFace Transformers"],
                "use_case": "Detects code smells, anti-patterns, style violations",
            },
            {
                "id": "security_detection",
                "name": "Security Vulnerability Detection",
                "architecture": "Ensemble: Random Forest + 1D CNN",
                "status": "production",
                "precision": 0.92,
                "recall": 0.85,
                "tech_stack": ["scikit-learn", "PyTorch"],
                "use_case": "Detects SQL injection, XSS, hardcoded secrets, etc.",
                "loaded": registry.security is not None,
            },
            {
                "id": "complexity_prediction",
                "name": "Code Complexity Prediction",
                "architecture": "XGBoost Regressor",
                "status": "production",
                "r2_score": 0.89,
                "tech_stack": ["XGBoost", "NumPy"],
                "use_case": "Predicts maintainability score (0–100)",
                "loaded": registry.complexity is not None,
            },
            {
                "id": "bug_predictor",
                "name": "Bug Prediction Model",
                "architecture": "Logistic Regression + PyTorch MLP",
                "status": "in_development",
                "expected_precision": "0.75-0.80",
                "tech_stack": ["scikit-learn", "PyTorch"],
                "use_case": "Predicts bug likelihood from static + git features",
                "loaded": registry.bug_predictor is not None,
            },
        ]
    }


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["analysis"])
def analyze_full(req: AnalyzeRequest):
    """
    Full analysis: runs security detection, complexity prediction,
    pattern recognition, and bug prediction.
    """
    t_start = time.perf_counter()
    source = req.code

    # --- Security ---
    if registry.security:
        sec_findings = registry.security.predict(source)
        sec_score = registry.security.vulnerability_score(source)
        security_out = {
            "findings": [vars(f) for f in sec_findings],
            "vulnerability_score": round(sec_score, 3),
            "summary": {
                "total": len(sec_findings),
                "critical": sum(1 for f in sec_findings if f.severity == "critical"),
                "high": sum(1 for f in sec_findings if f.severity == "high"),
                "medium": sum(1 for f in sec_findings if f.severity == "medium"),
                "low": sum(1 for f in sec_findings if f.severity == "low"),
            },
        }
    else:
        scan = scan_security_patterns(source)
        security_out = scan.to_dict()

    # --- Complexity ---
    if registry.complexity:
        complexity_result = registry.complexity.predict(source)
    else:
        from models.complexity_prediction import ComplexityPredictionModel as CPM
        complexity_result = CPM().predict(source)

    complexity_out = ComplexityOut(**complexity_result.to_dict())

    # --- Bug Prediction ---
    git_meta = None
    if req.git_metadata:
        git_meta = GitMetadata(**{
            k: req.git_metadata.get(k, 0)
            for k in ("code_churn", "author_count", "file_age_days",
                      "n_past_bugs", "commit_freq")
        })

    if registry.bug_predictor:
        bug_result = registry.bug_predictor.predict(source, git_meta)
    else:
        from models.bug_predictor import BugPredictionModel as BPM
        bug_result = BPM().predict(source, git_meta)

    bug_out = BugPredictionOut(**bug_result.to_dict())

    # --- Pattern Recognition (lazy load) ---
    pattern_out = None
    if registry.pattern_model is not None:
        try:
            pred = registry.pattern_model.predict(source)
            pattern_out = PatternOut(
                label=pred.label,
                confidence=pred.confidence,
                all_scores=pred.all_scores,
            )
        except Exception:
            pass

    # --- Overall score ---
    sec_penalty = security_out.get("summary", {}).get("critical", 0) * 15 + \
                  security_out.get("summary", {}).get("high", 0) * 8
    overall_score = max(0, int(complexity_out.score - sec_penalty))

    # Status
    crit = security_out.get("summary", {}).get("critical", 0)
    high = security_out.get("summary", {}).get("high", 0)
    if crit > 0:
        status = "critical"
    elif high > 0 or complexity_out.score < 60:
        status = "action_required"
    else:
        status = "clean"

    # Summary
    total_issues = security_out.get("summary", {}).get("total", 0)
    summary = (
        f"Found {total_issues} security issue(s). "
        f"Code quality score: {complexity_out.score}/100 ({complexity_out.grade}). "
        f"Bug risk: {bug_out.risk_level}."
    )

    duration = round(time.perf_counter() - t_start, 3)

    return FullAnalysisResponse(
        filename=req.filename,
        language=req.language,
        duration_seconds=duration,
        security=security_out,
        complexity=complexity_out,
        bug_prediction=bug_out,
        patterns=pattern_out,
        overall_score=overall_score,
        status=status,
        summary=summary,
    )


@app.post("/analyze/security", tags=["analysis"])
def analyze_security(req: AnalyzeRequest):
    """Security vulnerability scan only."""
    t_start = time.perf_counter()

    if registry.security:
        findings = registry.security.predict(req.code)
        vuln_score = registry.security.vulnerability_score(req.code)
        result = {
            "findings": [vars(f) for f in findings],
            "vulnerability_score": round(vuln_score, 3),
        }
    else:
        scan = scan_security_patterns(req.code)
        result = scan.to_dict()

    result["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return result


@app.post("/analyze/complexity", tags=["analysis"])
def analyze_complexity(req: AnalyzeRequest):
    """Complexity and maintainability analysis only."""
    t_start = time.perf_counter()

    if registry.complexity:
        result = registry.complexity.predict(req.code)
    else:
        from models.complexity_prediction import ComplexityPredictionModel as CPM
        result = CPM().predict(req.code)

    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


@app.post("/analyze/patterns", tags=["analysis"])
def analyze_patterns(req: AnalyzeRequest):
    """Pattern recognition (code smells / anti-patterns)."""
    t_start = time.perf_counter()

    if registry.pattern_model is None:
        # Lazy-load CodeBERT on first request
        try:
            from models.pattern_recognition import PatternRecognitionModel
            registry.pattern_model = PatternRecognitionModel(
                checkpoint_path="checkpoints/pattern"
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Pattern model unavailable: {e}"
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

    git_meta = None
    if req.git_metadata:
        git_meta = GitMetadata(**{
            k: req.git_metadata.get(k, 0)
            for k in ("code_churn", "author_count", "file_age_days",
                      "n_past_bugs", "commit_freq")
        })

    if registry.bug_predictor:
        result = registry.bug_predictor.predict(req.code, git_meta)
    else:
        from models.bug_predictor import BugPredictionModel as BPM
        result = BPM().predict(req.code, git_meta)

    out = result.to_dict()
    out["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    return out


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
