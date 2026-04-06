"""
Language adapter: routes metric computation to the correct language backend.

Supported languages: python, javascript, typescript, java, go, rust,
                     csharp, ruby, php, c, cpp
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
    ".go":   "go",
    ".rs":   "rust",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".php":  "php",
    ".c":    "c",
    ".h":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".kt":   "kotlin",
    ".kts":  "kotlin",
    ".swift":"swift",
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

    if lang == "go":
        from features.multi_lang.lang_go import compute_go_metrics
        return compute_go_metrics(source)

    if lang == "rust":
        from features.multi_lang.lang_rust import compute_rust_metrics
        return compute_rust_metrics(source)

    if lang == "csharp":
        from features.multi_lang.lang_csharp import compute_csharp_metrics
        return compute_csharp_metrics(source)

    if lang == "ruby":
        from features.multi_lang.lang_ruby import compute_ruby_metrics
        return compute_ruby_metrics(source)

    if lang == "php":
        from features.multi_lang.lang_php import compute_php_metrics
        return compute_php_metrics(source)

    if lang == "c":
        from features.multi_lang.lang_c import compute_c_metrics
        return compute_c_metrics(source)

    if lang == "cpp":
        from features.multi_lang.lang_c import compute_cpp_metrics
        return compute_cpp_metrics(source)

    # Kotlin / Swift: no tree-sitter package yet — use JS adapter as structural proxy
    if lang in ("kotlin", "swift"):
        from features.multi_lang.lang_js_ts import compute_js_metrics
        return compute_js_metrics(source, "javascript")

    # Unknown language — try Python parser, fall back to JS-based LOC analysis
    try:
        from features.code_metrics import compute_all_metrics
        return compute_all_metrics(source)
    except Exception:
        from features.multi_lang.lang_js_ts import compute_js_metrics
        return compute_js_metrics(source, "javascript")
