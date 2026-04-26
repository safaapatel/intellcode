/**
 * reviewHistory.ts
 * Persists analysis results in localStorage so Reviews, Dashboard,
 * and ReviewDetail can all read real data.
 */

import type { FullAnalysisResult } from "@/types/analysis";
import { STORAGE_KEYS } from "@/constants/storage";

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
  repoName?: string;            // e.g. "safaapatel/intellcode" — undefined for manual submissions
  modelVersion?: string;        // e.g. "1.1.0" — backend model version at time of analysis
}

const KEY = STORAGE_KEYS.reviewHistory;
const MAX_ENTRIES = 500;

function load(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

let _quotaWarned = false;

function save(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
    _quotaWarned = false; // reset if write succeeds
  } catch (e) {
    if (e instanceof DOMException && (e.name === "QuotaExceededError" || e.code === 22)) {
      // Storage full — trim to oldest half and retry once
      const trimmed = entries.slice(0, Math.floor(entries.length / 2));
      try {
        localStorage.setItem(KEY, JSON.stringify(trimmed));
        if (!_quotaWarned) {
          _quotaWarned = true;
          console.warn(
            "IntelliCode: localStorage quota exceeded — oldest review history entries were removed. " +
            "Consider clearing history from the Dashboard to free space."
          );
        }
      } catch {
        // Both attempts failed — history is not persisted this save
        console.error("IntelliCode: could not save review history (localStorage full).");
      }
    }
  }
}

/** Derive the top severity from a result */
function topSeverity(result: FullAnalysisResult): HistoryEntry["severity"] {
  const secFindings = result.security?.findings ?? [];
  if (secFindings.some((f) => f.severity === "critical")) return "critical";
  if (result.bug_prediction?.risk_level === "critical") return "critical";
  if (secFindings.some((f) => f.severity === "high")) return "high";
  if (result.bug_prediction?.risk_level === "high") return "high";
  if (secFindings.some((f) => f.severity === "medium")) return "medium";
  if (result.bug_prediction?.risk_level === "medium") return "medium";
  if (secFindings.length > 0) return "low";
  return "none";
}

/** Count total issues across all models */
function countIssues(result: FullAnalysisResult): number {
  const bugRisk = result.bug_prediction?.risk_level;
  const bugCount = bugRisk === "critical" || bugRisk === "high" ? 1 : 0;
  const patternCount = result.patterns?.label && result.patterns.label !== "clean" ? 1 : 0;
  return (
    (result.security?.findings?.length ?? 0) +
    (result.dead_code?.issues?.length ?? 0) +
    (result.refactoring?.suggestions?.length ?? 0) +
    (result.performance?.issues?.length ?? 0) +
    (result.dependencies?.issues?.length ?? 0) +
    (result.clones?.clones?.length ?? 0) +
    bugCount +
    patternCount
  );
}

/** Derive an overall score (0–100) from the backend's weighted score if available */
function overallScore(result: FullAnalysisResult): number {
  // Prefer the backend's pre-computed weighted score
  if (result.overall_score != null && result.overall_score > 0) return result.overall_score;
  // Fallback: average available model scores
  const scores: number[] = [];
  if (result.complexity?.score != null) scores.push(result.complexity.score);
  if (result.readability?.overall_score != null) scores.push(result.readability.overall_score);
  if (result.docs?.average_quality != null) scores.push(result.docs.average_quality);
  if (result.security?.vulnerability_score != null)
    scores.push(Math.max(0, 100 - result.security.vulnerability_score * 10));
  if (scores.length === 0) return 0;
  return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function addEntry(
  result: FullAnalysisResult,
  filename: string,
  language: string,
  repoName?: string
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
    repoName,
    modelVersion: result.model_version,
  };
  const combined = [entry, ...entries];
  let updated = combined;
  if (combined.length > MAX_ENTRIES) {
    // Evict oldest entries from the same repo first before touching other repos
    const sameRepo = combined.filter((e) => e.repoName === repoName);
    const otherRepos = combined.filter((e) => e.repoName !== repoName);
    const maxSameRepo = Math.max(200, MAX_ENTRIES - otherRepos.length);
    updated = [...sameRepo.slice(0, maxSameRepo), ...otherRepos].slice(0, MAX_ENTRIES);
  }
  save(updated);
  return entry;
}

export function getEntries(): HistoryEntry[] {
  return load();
}

/**
 * Returns the 15-dim static feature vector from the most recent previous
 * analysis of the same filename, or null if none exists.
 * Used by DRE to compute delta features between consecutive submissions.
 */
export function getPrevFeatures(filename: string): number[] | null {
  const entries = load();
  const prev = entries.find((e) => e.filename === filename && e.result?.complexity);
  if (!prev) return null;
  const c = prev.result.complexity;
  const bd = c.breakdown ?? {};
  // 15-dim — matches STATIC_FEAT_KEYS order in DRE training
  return [
    c.cyclomatic          ?? 0,
    c.cognitive           ?? 0,
    bd.max_function_cc    ?? 0,
    bd.avg_function_cc    ?? 0,
    c.sloc                ?? 0,
    bd.comments           ?? 0,
    bd.blank_lines        ?? 0,
    bd.halstead_volume    ?? 0,
    bd.halstead_difficulty ?? 0,
    bd.halstead_effort    ?? 0,
    c.halstead_bugs       ?? 0,
    c.n_long_functions    ?? 0,
    c.n_complex_functions ?? 0,
    bd.max_line_length    ?? 0,
    c.n_lines_over_80     ?? 0,
  ];
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

export function deleteEntry(id: string) {
  const entries = load().filter((e) => e.id !== id);
  save(entries);
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
