import type { FullAnalysisResult } from "@/types/analysis";
import { addEntry } from "@/services/reviewHistory";

const BASE_URL = import.meta.env.VITE_API_URL || "https://intellcode.onrender.com";

function notifyIfCritical(result: FullAnalysisResult) {
  if (!("Notification" in window)) return;
  const critical = result.security?.summary?.critical ?? 0;
  if (critical === 0) return;
  const send = () =>
    new Notification("⚠️ Critical Security Issue", {
      body: `${critical} critical finding${critical > 1 ? "s" : ""} in ${result.filename}`,
      icon: "/favicon.ico",
    });
  if (Notification.permission === "granted") {
    send();
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission().then((p) => { if (p === "granted") send(); });
  }
}

interface AnalyzeRequest {
  code: string;
  filename?: string;
  language?: string;
  git_metadata?: Record<string, number>;
}

// Tracks any in-flight analysis request so rapid resubmits cancel the previous one
let _currentAnalysisController: AbortController | null = null;

async function post<T>(path: string, body: AnalyzeRequest, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    if (res.status === 429) throw new Error("Rate limited — maximum 20 analyses per minute. Please wait and try again.");
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Run all ML models on the given code via a single /analyze call.
 * The backend now returns docs/performance/dependencies/readability
 * in wave 2 of the same request — no extra round trips needed.
 * Rapid resubmits automatically cancel the previous in-flight request.
 */
export async function analyzeCode(
  code: string,
  filename = "snippet.py",
  language = "python",
  gitMetadata?: Record<string, number>,
  repoName?: string
): Promise<FullAnalysisResult & { _entryId?: string }> {
  // Cancel any previous in-flight request
  if (_currentAnalysisController) {
    _currentAnalysisController.abort();
  }
  _currentAnalysisController = new AbortController();
  const { signal } = _currentAnalysisController;

  const req: AnalyzeRequest = { code, filename, language, git_metadata: gitMetadata };

  const result = await post<FullAnalysisResult>("/analyze", req, signal);

  _currentAnalysisController = null;

  // Persist to localStorage so Reviews + Dashboard show real data
  const entry = addEntry(result, filename, language, repoName);
  notifyIfCritical(result);

  return { ...result, _entryId: entry.id };
}

/**
 * Streaming analysis via /analyze/stream (SSE).
 * Calls onStep(stepName, progress) as each model completes.
 * Returns the full merged FullAnalysisResult on completion.
 */
export async function analyzeCodeStream(
  code: string,
  filename = "snippet.py",
  language = "python",
  onStep: (step: string, progress: number) => void,
  gitMetadata?: Record<string, number>,
  repoName?: string
): Promise<FullAnalysisResult & { _entryId?: string }> {
  const req: AnalyzeRequest = { code, filename, language, git_metadata: gitMetadata };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);

  const res = await fetch(`${BASE_URL}/analyze/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: controller.signal,
  });

  if (!res.ok) {
    clearTimeout(timeoutId);
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  if (!res.body) throw new Error("Response body is empty — streaming not supported");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE lines
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const payload = JSON.parse(line.slice(6));
          if (payload.status === "complete") {
            clearTimeout(timeoutId);
            const result = payload.result as FullAnalysisResult;
            const entry = addEntry(result, filename, language, repoName);
            notifyIfCritical(result);
            return { ...result, _entryId: entry.id };
          }
          if (payload.step && payload.progress != null) {
            onStep(payload.step, payload.progress);
          }
        } catch {
          // malformed line, skip
        }
      }
    }
  } finally {
    clearTimeout(timeoutId);
  }

  throw new Error("Stream ended without a complete result");
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error(`Models fetch failed: ${res.status}`);
  return res.json();
}

export async function getStats() {
  const res = await fetch(`${BASE_URL}/stats`);
  if (!res.ok) throw new Error(`Stats fetch failed: ${res.status}`);
  return res.json() as Promise<{
    total_analyses: number;
    avg_score: number | null;
    min_score: number | null;
    max_score: number | null;
    total_security_findings: number;
    specialist_cache_entries: number;
    models_ready: number;
  }>;
}

export async function clearCache() {
  const res = await fetch(`${BASE_URL}/cache/clear`, { method: "POST" });
  if (!res.ok) throw new Error(`Cache clear failed: ${res.status}`);
  return res.json();
}

export async function submitAnalysisFeedback(payload: {
  filename: string;
  overall_score: number;
  rating: "positive" | "negative";
  comment?: string;
}) {
  // Backend schema: finding_key + verdict (analysis-level uses "analysis_overall")
  const res = await fetch(`${BASE_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_id: payload.filename,
      finding_key: "analysis_overall",
      verdict: payload.rating === "positive" ? "helpful" : "not_helpful",
      comment: payload.comment,
    }),
  });
  if (!res.ok) throw new Error(`Feedback failed: ${res.status}`);
  return res.json();
}

/**
 * Quick analysis via /analyze/quick — security patterns + complexity + readability only.
 * Much faster than full analysis (~2-4s vs ~15s). Persists to history like full analysis.
 */
export async function analyzeCodeQuick(
  code: string,
  filename = "snippet.py",
  language = "python",
  repoName?: string
): Promise<FullAnalysisResult & { _entryId?: string }> {
  const req: AnalyzeRequest = { code, filename, language };
  const result = await post<FullAnalysisResult>("/analyze/quick", req);
  const entry = addEntry(result, filename, language, repoName);
  notifyIfCritical(result);
  return { ...result, _entryId: entry.id };
}

// ── Batch analysis ──────────────────────────────────────────────────────────

export interface BatchFileInput {
  code: string;
  filename: string;
  language?: string;
}

export interface BatchAnalysisResult {
  aggregate: {
    files_analysed: number;
    files_errored: number;
    avg_score: number;
    min_score: number;
    max_score: number;
    total_security_findings: number;
    critical_files: string[];
  };
  results: (FullAnalysisResult & { filename: string })[];
  errors: { error: string }[];
}

export async function analyzeBatch(
  files: BatchFileInput[],
  failFast = false
): Promise<BatchAnalysisResult> {
  const res = await fetch(`${BASE_URL}/analyze/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files, fail_fast: failFast }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Batch API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<BatchAnalysisResult>;
}

// ── Explain ─────────────────────────────────────────────────────────────────

export interface ExplainResult {
  features: Record<string, number>;
  models: {
    complexity?: {
      predicted_cognitive_complexity: number;
      score: number;
      top_features: { feature: string; value: number; importance: number; impact: string }[];
      interpretation: string;
    };
    security?: {
      pattern_findings: number;
      finding_categories: Record<string, number>;
      rf_features: Record<string, number>;
      high_risk_signals: string[];
      interpretation: string;
    };
    bug?: {
      bug_probability: number;
      risk_level: string;
      risk_drivers: { feature: string; value: number; threshold: number; exceeded: boolean; explanation: string }[];
      risk_factors: string[];
      interpretation: string;
    };
  };
  actionable_summary: string[];
}

export async function explainAnalysis(
  code: string,
  filename = "snippet.py",
  language = "python"
): Promise<ExplainResult> {
  const res = await fetch(`${BASE_URL}/analyze/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, filename, language }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Explain API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<ExplainResult>;
}

export async function getFeedbackStats(): Promise<{
  total: number;
  positive: number;
  negative: number;
  positive_rate: number;
}> {
  const res = await fetch(`${BASE_URL}/feedback/stats`);
  if (!res.ok) throw new Error(`Feedback stats failed: ${res.status}`);
  const raw = await res.json() as {
    total: number;
    helpful: number;
    not_helpful: number;
    false_positive: number;
  };
  const positive = raw.helpful ?? 0;
  const negative = (raw.not_helpful ?? 0) + (raw.false_positive ?? 0);
  return {
    total: raw.total,
    positive,
    negative,
    positive_rate: raw.total > 0 ? positive / raw.total : 0,
  };
}
