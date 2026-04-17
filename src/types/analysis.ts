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
  // Taint-flow enrichment (added by context_analyzer)
  argument_name?: string;
  taint_source?: string;
  taint_path?: string;
  user_controlled?: boolean;
  false_positive?: boolean;
  exploitability?: number;
  context_sentence?: string;
}

export interface ConformalInterval {
  lower: number;
  upper: number;
  point_estimate: number;
  coverage_level: number;  // e.g. 0.90
  quantile: number;
  n_cal: number;
  is_fallback: boolean;
}

export interface OODInfo {
  raw_probability: number;
  low_confidence: boolean;
  confidence_factor: number;
  sigma_distance: number;
}

export interface SecurityResult {
  findings: SecurityFinding[];
  false_positives?: SecurityFinding[];
  taint_enriched?: boolean;
  vulnerability_score: number;
  summary: {
    total: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  ood?: OODInfo;
  low_confidence?: boolean;
  low_confidence_reason?: string;
  conformal_interval?: ConformalInterval;
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
  risk_level: "low" | "medium" | "high" | "critical" | "uncertain";
  risk_factors: string[];
  confidence: number;
  static_score: number;
  git_score: number | null;
  abstained?: boolean;
  ood?: OODInfo;
  low_confidence?: boolean;
  low_confidence_reason?: string;
  probability_adjusted?: number;
  conformal_interval?: ConformalInterval;
  top_feature_importances?: { feature: string; importance: number }[];
  reliability_context?: {
    temporal_auc: number;
    in_dist_auc: number;
    lopo_auc: number;
    temporal_cv_auc: number;
    recommended_use: string;
  };
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
  // Novel model results
  function_risk?: FunctionRiskResult;
  grammar_anomaly?: GrammarAnomalyResult;
  dre?: DREResult;
  // Attached client-side before storing — not from backend
  code?: string;
}

export interface DREResult {
  risk_score: number;
  delta_contribution: number;
  top_delta_features: { feature: string; delta: number; current: number }[];
  confidence: number;
  static_score: number;
  model_type: string;
}

export interface FunctionRisk {
  name: string;
  lineno: number;
  end_lineno: number;
  raw_score: number;
  complexity_weight: number;
  attributed_score: number;
  cognitive_complexity: number;
  cyclomatic_complexity: number;
  sloc: number;
  rank: number;
  risk_level: "low" | "medium" | "high" | "critical";
  reason: string;
}

export interface FunctionRiskResult {
  functions: FunctionRisk[];
  top_k: FunctionRisk[];
  file_risk_score: number;
  total_functions: number;
  n_high_risk: number;
  localization_available: boolean;
}

export interface GrammarAnomalyResult {
  perplexity: number;
  anomaly_score: number;
  is_anomalous: boolean;
  top_anomalous_ngrams: { ngram: string; log_prob: number }[];
  n_tokens: number;
  grammar_coverage: number;
}
