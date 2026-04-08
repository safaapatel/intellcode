// ─── Security ────────────────────────────────────────────────────────────────

export interface SecurityFinding {
  vuln_type: string;
  severity: "critical" | "high" | "medium" | "low";
  title: string;
  description: string;
  lineno: number;
  snippet: string;
  confidence: number;
  cwe: string;
  source: string;
  decision?: Decision;  // injected by backend decision layer
}

export interface SecurityResult {
  findings: SecurityFinding[];
  vulnerability_score: number;
  summary: {
    total: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

// ─── Complexity ───────────────────────────────────────────────────────────────

export interface ComplexityResult {
  score: number;
  grade: string;
  cyclomatic: number;
  cognitive: number;
  halstead_bugs: number;
  maintainability_index: number;
  sloc: number;
  n_long_functions: number;
  n_complex_functions: number;
  n_lines_over_80: number;
  breakdown: Record<string, number>;
  function_issues: Array<{ name: string; issue: string; value: number }>;
}

// ─── Bug Prediction ───────────────────────────────────────────────────────────

export interface BugPredictionResult {
  bug_probability: number;
  risk_level: "low" | "medium" | "high" | "critical";
  risk_factors: string[];
  confidence: number;
  static_score: number;
  git_score: number | null;
}

// ─── Pattern Recognition ──────────────────────────────────────────────────────

export interface PatternResult {
  label: string;
  confidence: number;
  all_scores: Record<string, number>;
}

// ─── Code Clones ─────────────────────────────────────────────────────────────

export interface ClonePair {
  block_a: string;
  block_b: string;
  start_line_a: number;
  start_line_b: number;
  clone_type: "type1" | "type2" | "type3";
  similarity: number;
  description: string;
}

export interface ClonesResult {
  clones: ClonePair[];
  total_blocks: number;
  clone_rate: number;
  duplication_score: number;
  summary: string;
}

// ─── Refactoring ──────────────────────────────────────────────────────────────

export interface RefactoringSuggestion {
  refactoring_type: string;
  title: string;
  description: string;
  location: string;
  start_line: number;
  end_line: number;
  effort_minutes: number;
  priority: "critical" | "high" | "medium" | "low";
}

export interface RefactoringResult {
  suggestions: RefactoringSuggestion[];
  total_effort_minutes: number;
  priority_counts: Record<string, number>;
  summary: string;
}

// ─── Dead Code ────────────────────────────────────────────────────────────────

export interface DeadCodeIssue {
  issue_type: string;
  title: string;
  description: string;
  location: string;
  start_line: number;
  end_line: number;
  severity: "warning" | "info";
  removable: boolean;
}

export interface DeadCodeResult {
  issues: DeadCodeIssue[];
  dead_line_count: number;
  total_lines: number;
  dead_ratio: number;
  summary: string;
}

// ─── Technical Debt ───────────────────────────────────────────────────────────

export interface DebtCategory {
  name: string;
  debt_minutes: number;
  rating: string;
  items: string[];
}

export interface DebtResult {
  total_debt_minutes: number;
  overall_rating: string;
  categories: DebtCategory[];
  interest_per_day: number;
  payoff_days: number;
  summary: string;
}

// ─── Documentation Quality ────────────────────────────────────────────────────

export interface DocIssue {
  issue_type: string;
  severity: string;
  symbol: string;
  location: string;
  start_line: number;
  message: string;
}

export interface SymbolDocScore {
  name: string;
  kind: string;
  start_line: number;
  has_docstring: boolean;
  quality_score: number;
  issues: string[];
}

export interface DocQualityResult {
  coverage: number;
  average_quality: number;
  grade: string;
  total_symbols: number;
  documented_symbols: number;
  symbol_scores: SymbolDocScore[];
  issues: DocIssue[];
  summary: string;
}

// ─── Performance ──────────────────────────────────────────────────────────────

export interface PerformanceIssue {
  pattern_type: string;
  title: string;
  description: string;
  location: string;
  start_line: number;
  severity: "high" | "medium" | "low";
  speedup_hint: string;
}

export interface PerformanceResult {
  issues: PerformanceIssue[];
  hotspot_score: number;
  severity_counts: Record<string, number>;
  summary: string;
}

// ─── Dependencies ─────────────────────────────────────────────────────────────

export interface ImportRecord {
  module: string;
  names: string[];
  is_relative: boolean;
  relative_level: number;
  is_wildcard: boolean;
  lineno: number;
  inside_function: boolean;
  category: "stdlib" | "external" | "local";
}

export interface CouplingIssue {
  issue_type: string;
  severity: string;
  title: string;
  description: string;
  location: string;
  start_line: number;
}

export interface DependencyResult {
  imports: ImportRecord[];
  fan_out: number;
  fan_out_stdlib: number;
  fan_out_external: number;
  fan_out_local: number;
  instability: number;
  coupling_score: number;
  issues: CouplingIssue[];
  dependency_map: Record<string, string[]>;
  summary: string;
}

// ─── Readability ──────────────────────────────────────────────────────────────

export interface ReadabilityIssue {
  dimension: string;
  title: string;
  description: string;
  location: string;
  start_line: number;
  penalty: number;
}

export interface DimensionScore {
  name: string;
  score: number;
  weight: number;
  issues: string[];
}

export interface ReadabilityResult {
  overall_score: number;
  grade: string;
  dimensions: DimensionScore[];
  issues: ReadabilityIssue[];
  top_improvements: string[];
  summary: string;
}

// ─── Full Combined Result ─────────────────────────────────────────────────────

// ─── Decision Layer ───────────────────────────────────────────────────────────

export type DecisionAction = "fix_now" | "review_manually" | "low_priority" | "unreliable";

export interface Decision {
  action: DecisionAction;
  label: string;
  explanation: string;
  priority: number;
  color: "red" | "yellow" | "blue" | "gray";
}

export interface TrustSummaryItem {
  model: string;
  reliability: "high" | "medium" | "low";
  message: string;
}

export interface TrustSummary {
  overall_reliable: boolean;
  language_in_distribution: boolean;
  items: TrustSummaryItem[];
}

// ─── Full Combined Result ─────────────────────────────────────────────────────

export interface FullAnalysisResult {
  filename: string;
  language: string;
  duration_seconds: number;
  security: SecurityResult;
  complexity: ComplexityResult;
  bug_prediction: BugPredictionResult;
  patterns: PatternResult | null;
  clones: ClonesResult;
  refactoring: RefactoringResult;
  dead_code: DeadCodeResult;
  technical_debt: DebtResult;
  overall_score: number;
  status: "clean" | "action_required" | "critical";
  summary: string;
  trust_summary?: TrustSummary;
  model_version?: string;
  // Added by frontend after parallel calls:
  docs?: DocQualityResult;
  performance?: PerformanceResult;
  dependencies?: DependencyResult;
  readability?: ReadabilityResult;
}
