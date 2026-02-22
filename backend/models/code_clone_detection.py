"""
Code Clone Detection
Detects duplicate and near-duplicate code blocks using embedding-based similarity.

Clone Types:
    Type-1  Exact copies (after whitespace/comment normalization)
    Type-2  Renamed variables / renamed identifiers
    Type-3  Near-miss clones (structurally similar with modifications)

Approach:
    1. Extract function/method blocks from source via AST
    2. Embed each block using TF-IDF token n-grams (fast, no GPU needed)
       OR sentence-transformers if available (higher quality)
    3. Compare pairwise using cosine similarity
    4. Optionally build a FAISS index for large-scale search across files

Architecture:
    CodeCloneDetector — main class
    ClonePair         — result dataclass for a detected clone pair
    CodeBlock         — represents one extracted code block
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    name: str               # function/class name or "module"
    source: str             # raw source text
    start_line: int
    end_line: int
    normalized: str = field(default="", repr=False)  # for Type-2 comparison
    fingerprint: str = field(default="", repr=False)  # MD5 for Type-1


@dataclass
class ClonePair:
    block_a: str            # name of block A
    block_b: str            # name of block B
    start_line_a: int
    start_line_b: int
    clone_type: str         # "type1" | "type2" | "type3"
    similarity: float       # 0.0 – 1.0
    description: str

    def to_dict(self) -> dict:
        return {
            "block_a": self.block_a,
            "block_b": self.block_b,
            "start_line_a": self.start_line_a,
            "start_line_b": self.start_line_b,
            "clone_type": self.clone_type,
            "similarity": round(self.similarity, 3),
            "description": self.description,
        }


@dataclass
class CloneDetectionResult:
    clones: list[ClonePair]
    total_blocks: int
    clone_rate: float           # fraction of blocks involved in at least one clone
    duplication_score: float    # 0 (no clones) → 100 (everything duplicated)
    summary: str

    def to_dict(self) -> dict:
        return {
            "clones": [c.to_dict() for c in self.clones],
            "total_blocks": self.total_blocks,
            "clone_rate": round(self.clone_rate, 3),
            "duplication_score": round(self.duplication_score, 1),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\b", re.IGNORECASE)


def _strip_comments(source: str) -> str:
    """Remove # comments and blank lines."""
    lines = []
    for line in source.splitlines():
        stripped = line.split("#")[0].rstrip()
        if stripped.strip():
            lines.append(stripped)
    return "\n".join(lines)


def _normalize_whitespace(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def _type1_normalize(source: str) -> str:
    """Normalize for Type-1 comparison: strip comments + collapse whitespace."""
    return _normalize_whitespace(_strip_comments(source))


def _type2_normalize(source: str) -> str:
    """
    Normalize for Type-2 comparison: replace all identifiers with a
    canonical token so renamed variables still match.
    """
    base = _type1_normalize(source)
    seen: dict[str, str] = {}
    counter = [0]

    def replace(m: re.Match) -> str:
        word = m.group(1)
        # Keep Python keywords and builtins as-is
        _KEEP = {
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "import", "from", "as", "with", "try", "except", "finally",
            "raise", "pass", "break", "continue", "lambda", "yield",
            "and", "or", "not", "in", "is", "None", "True", "False",
            "self", "cls", "print", "len", "range", "int", "str", "float",
            "list", "dict", "set", "tuple", "bool", "type", "isinstance",
        }
        if word in _KEEP:
            return word
        if word not in seen:
            counter[0] += 1
            seen[word] = f"VAR{counter[0]}"
        return seen[word]

    return _ID_PATTERN.sub(replace, base)


def _fingerprint(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def _extract_blocks(source: str) -> list[CodeBlock]:
    """Extract top-level and nested functions/classes as CodeBlocks."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    blocks: list[CodeBlock] = []

    def get_source(node) -> str:
        start = node.lineno - 1
        end = node.end_lineno
        snippet = textwrap.dedent("\n".join(lines[start:end]))
        return snippet

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not hasattr(node, "end_lineno"):
                continue
            src = get_source(node)
            if len(src.strip().splitlines()) < 3:
                continue  # skip trivial 1-2 line blocks
            t1 = _type1_normalize(src)
            t2 = _type2_normalize(src)
            blocks.append(CodeBlock(
                name=node.name,
                source=src,
                start_line=node.lineno,
                end_line=node.end_lineno,
                normalized=t2,
                fingerprint=_fingerprint(t1),
            ))

    return blocks


# ---------------------------------------------------------------------------
# Embedding + similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _TFIDFEmbedder:
    """
    Lightweight TF-IDF token n-gram embedder.
    No external ML libraries required.
    """

    def __init__(self, max_features: int = 500, ngram_range: tuple = (1, 2)):
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        result = list(tokens)
        if self._ngram_range[1] >= 2:
            result += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return result

    def fit(self, texts: list[str]):
        import math
        n = len(texts)
        df: dict[str, int] = {}
        tf_all = []
        for text in texts:
            toks = set(self._tokenize(text))
            tf_all.append(toks)
            for t in toks:
                df[t] = df.get(t, 0) + 1

        # Keep top max_features by df
        sorted_vocab = sorted(df.keys(), key=lambda t: -df[t])[:self._max_features]
        self._vocab = {t: i for i, t in enumerate(sorted_vocab)}
        self._idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in sorted_vocab}
        self._fitted = True

    def transform(self, text: str) -> list[float]:
        toks = self._tokenize(text)
        freq: dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(toks), 1)
        vec = [0.0] * len(self._vocab)
        for t, idx in self._vocab.items():
            if t in freq:
                tf = freq[t] / total
                vec[idx] = tf * self._idf.get(t, 1.0)
        return vec

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        self.fit(texts)
        return [self.transform(t) for t in texts]


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

TYPE1_THRESHOLD = 0.98    # essentially identical
TYPE2_THRESHOLD = 0.85    # renamed identifiers
TYPE3_THRESHOLD = 0.70    # structurally similar


class CodeCloneDetector:
    """
    Detects Type-1, Type-2, and Type-3 code clones within a source file
    or across multiple files.

    Usage:
        detector = CodeCloneDetector()
        result = detector.detect(source_code)
    """

    def __init__(self, type3_threshold: float = TYPE3_THRESHOLD):
        self._threshold = type3_threshold
        self._embedder = _TFIDFEmbedder()

    def detect(self, source: str) -> CloneDetectionResult:
        """
        Detect clones in a single source file.

        Args:
            source: Python source code string.

        Returns:
            CloneDetectionResult with all clone pairs found.
        """
        blocks = _extract_blocks(source)
        if len(blocks) < 2:
            return CloneDetectionResult(
                clones=[],
                total_blocks=len(blocks),
                clone_rate=0.0,
                duplication_score=0.0,
                summary="Not enough code blocks to compare.",
            )

        clones = self._find_clones(blocks)
        involved = set()
        for c in clones:
            involved.add(c.block_a)
            involved.add(c.block_b)

        clone_rate = len(involved) / len(blocks)
        duplication_score = min(100.0, clone_rate * 100 * 1.5)

        n = len(clones)
        t1 = sum(1 for c in clones if c.clone_type == "type1")
        t2 = sum(1 for c in clones if c.clone_type == "type2")
        t3 = sum(1 for c in clones if c.clone_type == "type3")

        if n == 0:
            summary = "No code clones detected."
        else:
            summary = (
                f"Found {n} clone pair(s): "
                f"{t1} exact (Type-1), {t2} renamed (Type-2), {t3} near-miss (Type-3). "
                f"{len(involved)}/{len(blocks)} blocks involved."
            )

        return CloneDetectionResult(
            clones=clones,
            total_blocks=len(blocks),
            clone_rate=clone_rate,
            duplication_score=duplication_score,
            summary=summary,
        )

    def detect_across(self, sources: dict[str, str]) -> CloneDetectionResult:
        """
        Detect clones across multiple files.

        Args:
            sources: {filename: source_code} mapping.

        Returns:
            CloneDetectionResult aggregated across all files.
        """
        all_blocks: list[CodeBlock] = []
        for filename, src in sources.items():
            for block in _extract_blocks(src):
                block.name = f"{filename}::{block.name}"
                all_blocks.append(block)

        if len(all_blocks) < 2:
            return CloneDetectionResult(
                clones=[],
                total_blocks=len(all_blocks),
                clone_rate=0.0,
                duplication_score=0.0,
                summary="Not enough blocks across files to compare.",
            )

        clones = self._find_clones(all_blocks)
        # Only keep cross-file clones
        clones = [c for c in clones
                  if c.block_a.split("::")[0] != c.block_b.split("::")[0]]

        involved = set()
        for c in clones:
            involved.add(c.block_a)
            involved.add(c.block_b)

        clone_rate = len(involved) / len(all_blocks) if all_blocks else 0.0
        duplication_score = min(100.0, clone_rate * 100 * 1.5)

        summary = (
            f"Cross-file analysis: {len(clones)} clone pair(s) "
            f"across {len(sources)} files."
        ) if clones else "No cross-file clones detected."

        return CloneDetectionResult(
            clones=clones,
            total_blocks=len(all_blocks),
            clone_rate=clone_rate,
            duplication_score=duplication_score,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_clones(self, blocks: list[CodeBlock]) -> list[ClonePair]:
        clones: list[ClonePair] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Type-1: exact fingerprint match
        fp_map: dict[str, list[CodeBlock]] = {}
        for b in blocks:
            fp_map.setdefault(b.fingerprint, []).append(b)
        for fp, group in fp_map.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ba, bb = group[i], group[j]
                    key = (min(ba.name, bb.name), max(ba.name, bb.name))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        clones.append(ClonePair(
                            block_a=ba.name,
                            block_b=bb.name,
                            start_line_a=ba.start_line,
                            start_line_b=bb.start_line,
                            clone_type="type1",
                            similarity=1.0,
                            description="Exact copy (after whitespace normalization)",
                        ))

        # Type-2: normalized identifier match
        norm_map: dict[str, list[CodeBlock]] = {}
        for b in blocks:
            norm_fp = _fingerprint(b.normalized)
            norm_map.setdefault(norm_fp, []).append(b)
        for norm_fp, group in norm_map.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ba, bb = group[i], group[j]
                    key = (min(ba.name, bb.name), max(ba.name, bb.name))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        clones.append(ClonePair(
                            block_a=ba.name,
                            block_b=bb.name,
                            start_line_a=ba.start_line,
                            start_line_b=bb.start_line,
                            clone_type="type2",
                            similarity=0.95,
                            description="Renamed-variable clone (same structure, different identifiers)",
                        ))

        # Type-3: embedding similarity
        texts = [b.normalized for b in blocks]
        try:
            vecs = self._embedder.fit_transform(texts)
        except Exception:
            return clones  # skip Type-3 if embedding fails

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                ba, bb = blocks[i], blocks[j]
                key = (min(ba.name, bb.name), max(ba.name, bb.name))
                if key in seen_pairs:
                    continue
                sim = _cosine_similarity(vecs[i], vecs[j])
                if sim >= self._threshold:
                    seen_pairs.add(key)
                    clones.append(ClonePair(
                        block_a=ba.name,
                        block_b=bb.name,
                        start_line_a=ba.start_line,
                        start_line_b=bb.start_line,
                        clone_type="type3",
                        similarity=sim,
                        description=f"Near-miss clone (structural similarity: {sim:.0%})",
                    ))

        clones.sort(key=lambda c: -c.similarity)
        return clones
