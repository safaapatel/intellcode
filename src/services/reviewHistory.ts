/**
 * reviewHistory.ts
 * Persists analysis results in localStorage so Reviews, Dashboard,
 * and ReviewDetail can all read real data.
 */

import type { FullAnalysisResult } from "@/types/analysis";

export interface HistoryEntry {
  id: string;
  filename: string;
  language: string;
  submittedAt: string;          // ISO string
  summary: string;
  overallScore: number;
  issueCount: number;
  severity: "critical" | "high" | "medium" | "low" | "none";
  status: "completed" | "in_progress" | "pending";
  result: FullAnalysisResult;
  resolvedIssues: string[];     // issue keys that user marked resolved
  falsePositives: string[];     // issue keys marked false positive
}

const KEY = "intellcode_review_history";
const MAX_ENTRIES = 50;

function load(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function save(entries: HistoryEntry[]) {
  localStorage.setItem(KEY, JSON.stringify(entries));
}

/** Derive the top severity from a result */
function topSeverity(result: FullAnalysisResult): HistoryEntry["severity"] {
  const secFindings = result.security?.findings ?? [];
  if (secFindings.some((f) => f.severity === "critical")) return "critical";
  if (result.bugs?.risk_level === "critical") return "critical";
  if (secFindings.some((f) => f.severity === "high")) return "high";
  if (result.bugs?.risk_level === "high") return "high";
  if (secFindings.some((f) => f.severity === "medium")) return "medium";
  if (result.bugs?.risk_level === "medium") return "medium";
  if (secFindings.length > 0) return "low";
  return "none";
}

/** Count total issues across all models */
function countIssues(result: FullAnalysisResult): number {
  return (
    (result.security?.findings?.length ?? 0) +
    (result.dead_code?.issues?.length ?? 0) +
    (result.refactoring?.suggestions?.length ?? 0) +
    (result.performance?.issues?.length ?? 0) +
    (result.dependencies?.issues?.length ?? 0) +
    (result.clones?.clones?.length ?? 0)
  );
}

/** Derive an overall score (0–100) from multiple model scores */
function overallScore(result: FullAnalysisResult): number {
  const scores: number[] = [];
  if (result.complexity?.score != null) scores.push(result.complexity.score);
  if (result.readability?.overall_score != null) scores.push(result.readability.overall_score);
  if (result.docs?.average_quality != null) scores.push(result.docs.average_quality);
  if (result.security?.vulnerability_score != null)
    scores.push(Math.max(0, 100 - result.security.vulnerability_score * 10));
  if (result.overall_score) scores.push(result.overall_score);
  if (scores.length === 0) return 75;
  return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function addEntry(
  result: FullAnalysisResult,
  filename: string,
  language: string
): HistoryEntry {
  const entries = load();
  const entry: HistoryEntry = {
    id: crypto.randomUUID(),
    filename,
    language,
    submittedAt: new Date().toISOString(),
    summary: result.summary ?? "Analysis complete",
    overallScore: overallScore(result),
    issueCount: countIssues(result),
    severity: topSeverity(result),
    status: "completed",
    result,
    resolvedIssues: [],
    falsePositives: [],
  };
  const updated = [entry, ...entries].slice(0, MAX_ENTRIES);
  save(updated);
  return entry;
}

export function getEntries(): HistoryEntry[] {
  return load();
}

export function getEntry(id: string): HistoryEntry | undefined {
  return load().find((e) => e.id === id);
}

export function markResolved(entryId: string, issueKey: string) {
  const entries = load();
  const e = entries.find((x) => x.id === entryId);
  if (!e) return;
  if (!e.resolvedIssues.includes(issueKey)) {
    e.resolvedIssues.push(issueKey);
  }
  // Remove from false positives if was there
  e.falsePositives = e.falsePositives.filter((k) => k !== issueKey);
  save(entries);
}

export function markFalsePositive(entryId: string, issueKey: string) {
  const entries = load();
  const e = entries.find((x) => x.id === entryId);
  if (!e) return;
  if (!e.falsePositives.includes(issueKey)) {
    e.falsePositives.push(issueKey);
  }
  e.resolvedIssues = e.resolvedIssues.filter((k) => k !== issueKey);
  save(entries);
}

export function clearHistory() {
  localStorage.removeItem(KEY);
}

/** Compute dashboard stats from stored history */
export function getDashboardStats() {
  const entries = load();
  if (entries.length === 0) {
    return null; // caller falls back to mock
  }
  const totalReviews = entries.length;
  const issuesFound = entries.reduce((s, e) => s + e.issueCount, 0);
  const avgScore = Math.round(
    entries.reduce((s, e) => s + e.overallScore, 0) / entries.length
  );
  // Week-over-week: entries in last 7 days vs prior 7
  const now = Date.now();
  const week = 7 * 24 * 60 * 60 * 1000;
  const thisWeek = entries.filter(
    (e) => now - new Date(e.submittedAt).getTime() < week
  ).length;
  return { totalReviews, issuesFound, avgScore, thisWeek };
}
