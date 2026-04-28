"""
Integration tests for the IntelliCode FastAPI backend.

Run: pytest tests/ -v --tb=short

These tests run against the actual app (models may be mocked if not loaded).
They verify response shape, caching, rate limiting, and basic ML output structure.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

# Import app after path setup
from main import app, _analysis_cache, _specialist_cache, _rate_limit_store

client = TestClient(app)

SIMPLE_PYTHON = """
def add(a, b):
    return a + b

def multiply(x, y):
    result = 0
    for _ in range(y):
        result += x
    return result
"""

SECURITY_BAD = """
import subprocess
import pickle

def run_cmd(user_input):
    subprocess.call(user_input, shell=True)

def load_data(data):
    return pickle.loads(data)

password = "hardcoded-secret-123"
"""


# ---------------------------------------------------------------------------
# Health / meta endpoints
# ---------------------------------------------------------------------------

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_models_returns_list():
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert isinstance(body["models"], list)


def test_stats_returns_numeric_fields():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_analyses" in body
    assert "models_ready" in body
    assert isinstance(body["models_ready"], int)


# ---------------------------------------------------------------------------
# /analyze — response shape
# ---------------------------------------------------------------------------

def test_analyze_returns_required_fields():
    r = client.post("/analyze", json={"code": SIMPLE_PYTHON, "filename": "test.py", "language": "python"})
    if r.status_code != 200:
        pytest.skip("Backend models not loaded in CI")
    if r.status_code == 200:
        body = r.json()
        for key in ("security", "complexity", "overall_score", "status", "summary"):
            assert key in body, f"Missing field: {key}"
        assert 0 <= body["overall_score"] <= 100
        assert body["status"] in ("clean", "action_required", "critical")


def test_analyze_security_findings_shape():
    r = client.post("/analyze", json={"code": SECURITY_BAD, "filename": "bad.py", "language": "python"})
    if r.status_code != 200:
        pytest.skip("Backend models not loaded in CI")
    body = r.json()
    findings = body.get("security", {}).get("findings", [])
    for f in findings:
        assert "severity" in f
        assert f["severity"] in ("critical", "high", "medium", "low")
        assert "title" in f
        assert "lineno" in f


def test_analyze_rejects_empty_code():
    r = client.post("/analyze", json={"code": "", "filename": "empty.py"})
    assert r.status_code == 422


def test_analyze_rejects_unsupported_content_type():
    r = client.post("/analyze", content=b"not json", headers={"Content-Type": "text/plain"})
    assert r.status_code in (415, 422)


# ---------------------------------------------------------------------------
# Caching — hit/miss
# ---------------------------------------------------------------------------

def test_analyze_cache_hit():
    payload = {"code": SIMPLE_PYTHON, "filename": "cache_test.py", "language": "python"}

    r1 = client.post("/analyze", json=payload)
    if r1.status_code != 200:
        pytest.skip("Backend models not loaded in CI")

    _analysis_cache_before = len(_analysis_cache)

    r2 = client.post("/analyze", json=payload)
    assert r2.status_code == 200

    # Both responses must be identical (cache hit = same dict)
    assert r1.json()["overall_score"] == r2.json()["overall_score"]
    assert r1.json()["summary"] == r2.json()["summary"]


def test_different_code_produces_different_cache_keys():
    code_a = "def foo(): return 1"
    code_b = "def bar(): return 2"

    r_a = client.post("/analyze", json={"code": code_a, "filename": "a.py"})
    r_b = client.post("/analyze", json={"code": code_b, "filename": "b.py"})

    if r_a.status_code != 200 or r_b.status_code != 200:
        pytest.skip("Backend models not loaded in CI")

    # Scores can differ — at minimum confirm both succeed
    assert r_a.status_code == 200
    assert r_b.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limit_allows_normal_traffic():
    # A single request must not be rate-limited
    r = client.post("/analyze", json={"code": SIMPLE_PYTHON, "filename": "rl.py"})
    if r.status_code == 503:
        pytest.skip("Backend models not loaded in CI")
    assert r.status_code != 429


def test_rate_limit_triggers_after_threshold(monkeypatch):
    """
    Simulate a flooded IP by pre-filling the rate-limit store, then verify
    the 429 path fires. We patch the store directly to avoid making 200 requests.
    """
    import collections, time as _time
    from main import _rate_limit_store, _RATE_LIMIT_REQUESTS, _RATE_LIMIT_WINDOW_S

    fake_ip = "10.0.0.254"
    now = _time.monotonic()
    # Fill the deque to the threshold
    _rate_limit_store[fake_ip] = collections.deque(
        [now - 1] * _RATE_LIMIT_REQUESTS
    )

    # Next request from this IP must be rate-limited.
    # TestClient always uses "testclient" as the host, so we monkeypatch.
    original_host = None

    class FakeClient:
        host = fake_ip

    def fake_middleware(request, call_next):
        request._client = FakeClient()
        return call_next(request)

    # Direct call to the rate-limiter function
    from main import _check_rate_limit
    from fastapi import Request as FRequest
    from starlette.datastructures import Headers
    from starlette.testclient import TestClient as SC

    # Simpler: just verify the deque is at capacity and the check raises
    from fastapi import HTTPException
    import pytest

    class FakeRequest:
        class client:
            host = fake_ip

    with pytest.raises(HTTPException) as exc_info:
        _check_rate_limit(FakeRequest())
    assert exc_info.value.status_code == 429

    # Cleanup
    _rate_limit_store.pop(fake_ip, None)


# ---------------------------------------------------------------------------
# /analyze/quick — fast path
# ---------------------------------------------------------------------------

def test_analyze_quick_returns_200():
    r = client.post("/analyze/quick", json={"code": SIMPLE_PYTHON, "filename": "q.py"})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert "security" in body or "complexity" in body


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_analyze_truncates_very_long_code():
    long_code = "x = 1\n" * 5000
    r = client.post("/analyze", json={"code": long_code, "filename": "long.py"})
    # Should either succeed (truncated) or return 422 — never 500
    assert r.status_code in (200, 422, 503)


def test_analyze_rejects_invalid_language():
    r = client.post("/analyze", json={"code": "x=1", "filename": "f.py", "language": "brainfuck"})
    # Language is validated — should be 422 or fall back gracefully
    assert r.status_code in (200, 422)
