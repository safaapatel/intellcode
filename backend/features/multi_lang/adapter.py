"""
Language adapter: routes metric computation to the correct language backend.

Supported languages: python, javascript, typescript, java
"""

import os

# Extension -> canonical language name
_EXT_MAP = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".java": "java",
}


def detect_language(filename: str, hint: str = "python") -> str:
    """Return canonical language name from filename extension, falling back to hint."""
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_MAP.get(ext, hint.lower())


def compute_metrics_for_language(source: str, language: str):
    """
    Compute CodeMetricsResult for *source* in the given *language*.
    Returns a CodeMetricsResult (same type as compute_all_metrics()).
    """
    lang = language.lower()

    if lang == "python":
        from features.code_metrics import compute_all_metrics
        return compute_all_metrics(source)

    if lang in ("javascript", "typescript"):
        from features.multi_lang.lang_js_ts import compute_js_metrics
        return compute_js_metrics(source, lang)

    if lang == "java":
        from features.multi_lang.lang_java import compute_java_metrics
        return compute_java_metrics(source)

    # Unrecognised language — try Python parser, fall back to basic LOC
    try:
        from features.code_metrics import compute_all_metrics
        return compute_all_metrics(source)
    except Exception:
        from features.multi_lang.lang_js_ts import compute_js_metrics
        return compute_js_metrics(source, "javascript")
