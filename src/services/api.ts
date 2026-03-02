import type {
  FullAnalysisResult,
  DocQualityResult,
  PerformanceResult,
  DependencyResult,
  ReadabilityResult,
} from "@/types/analysis";
import { addEntry } from "@/services/reviewHistory";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface AnalyzeRequest {
  code: string;
  filename?: string;
  language?: string;
  git_metadata?: Record<string, number>;
}

async function post<T>(path: string, body: AnalyzeRequest): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/** Run all 12 ML models on the given code. Calls /analyze + 4 specialist endpoints in parallel. */
export async function analyzeCode(
  code: string,
  filename = "snippet.py",
  language = "python",
  gitMetadata?: Record<string, number>
): Promise<FullAnalysisResult> {
  const req: AnalyzeRequest = {
    code,
    filename,
    language,
    git_metadata: gitMetadata,
  };

  // Core analysis (8 models) + 4 specialist models — all in parallel
  const [core, docs, performance, dependencies, readability] = await Promise.all([
    post<FullAnalysisResult>("/analyze", req),
    post<DocQualityResult>("/analyze/docs", req).catch(() => undefined),
    post<PerformanceResult>("/analyze/performance", req).catch(() => undefined),
    post<DependencyResult>("/analyze/dependencies", req).catch(() => undefined),
    post<ReadabilityResult>("/analyze/readability", req).catch(() => undefined),
  ]);

  const result: FullAnalysisResult = {
    ...core,
    docs,
    performance,
    dependencies,
    readability,
  };

  // Persist to localStorage so Reviews + Dashboard show real data
  addEntry(result, filename, language);

  return result;
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${BASE_URL}/models`);
  return res.json();
}
