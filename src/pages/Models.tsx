import { useState, useEffect } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  Shield,
  BarChart2,
  Bug,
  Copy,
  Wrench,
  FileX,
  CreditCard,
  BookOpen,
  Gauge,
  Network,
  Eye,
  CheckCircle2,
  Database,
  Layers,
  TrendingUp,
  Cpu,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Activity,
  AlertTriangle,
} from "lucide-react";
import { getALStatus, getModelsEvaluation, type ALStatus, type ModelsEvaluation, type LOPOResult } from "@/services/api";

// ─── All models with real checkpoint metrics ───────────────────────────────

const ALL_MODELS = [
  {
    id: "pattern",
    name: "Pattern Recognition",
    icon: Brain,
    iconColor: "text-primary",
    iconBg: "bg-primary/10",
    architecture: "Random Forest (300 trees, calibrated with Platt scaling)",
    trainingData: "1 825 Python functions labelled by code-quality heuristics",
    useCase: "Detects: clean, code_smell, anti_pattern, style_violation",
    techStack: ["scikit-learn", "Python AST", "radon"],
    metrics: [
      { label: "Test Accuracy",   value: "86.1%", pct: 86 },
      { label: "Test AUC (OvR)", value: "0.969",  pct: 97 },
      { label: "CV F1 (macro)",  value: "0.797",  pct: 80 },
      { label: "Train / Test",   value: "1 551 / 274", pct: 85 },
    ],
    description:
      "Lightweight Random Forest trained on 26 static code-metric and AST features. Four-class classifier (clean / code_smell / anti_pattern / style_violation) with calibrated probabilities — no GPU or model download required.",
    endpoint: "POST /analyze/patterns",
  },
  {
    id: "security",
    name: "Security Detection",
    icon: Shield,
    iconColor: "text-red-400",
    iconBg: "bg-red-500/10",
    architecture: "RF + 1D CNN + Contrastive encoder (NT-Xent pre-trained, 3-component ensemble)",
    trainingData: "Bandit-labelled examples + CVE security-fix commits + CVEFixes (vulnerable/fixed pairs)",
    useCase: "SQL injection, XSS, hardcoded secrets, path traversal",
    techStack: ["scikit-learn", "PyTorch", "NT-Xent", "Python"],
    metrics: [
      { label: "Ensemble AUC",        value: "0.928", pct: 93 },
      { label: "RF Threshold",        value: "0.417", pct: 52 },
      { label: "Vuln Recall",         value: "54%",   pct: 54 },
      { label: "Contrastive (target)", value: "0.65+ LOPO", pct: 65 },
    ],
    description:
      "Three-component ensemble: Random Forest (50%) + 1D CNN (35%) + NT-Xent contrastive encoder (15%). The contrastive component is pre-trained on (vulnerable, fixed) pairs from CVEFixes using SimCLR-style loss, targeting LOPO AUC improvement from 0.494 to 0.65+. Youden-tuned threshold (0.417) improves vulnerable recall from 42% to 54%.",
    endpoint: "POST /analyze/security",
  },
  {
    id: "complexity",
    name: "Complexity Prediction",
    icon: BarChart2,
    iconColor: "text-yellow-400",
    iconBg: "bg-yellow-500/10",
    architecture: "XGBoost regressor (500 trees, depth 6)",
    trainingData: "Cyclomatic complexity, Halstead, LOC — 1 500 samples",
    useCase: "Maintainability score, refactoring priority",
    techStack: ["XGBoost", "NumPy", "Python"],
    metrics: [
      { label: "Test R²",       value: "0.692", pct: 69 },
      { label: "Test RMSE",     value: "23.39", pct: 55 },
      { label: "Spearman rho",  value: "0.865", pct: 87 },
      { label: "Multi-seed R²", value: "0.685", pct: 69 },
    ],
    description:
      "XGBoost regressor predicting cognitive complexity from Halstead volume, cyclomatic complexity, SLOC, and 12 other code metrics.",
    endpoint: "POST /analyze/complexity",
  },
  {
    id: "bugs",
    name: "Bug Prediction",
    icon: Bug,
    iconColor: "text-orange-400",
    iconBg: "bg-orange-500/10",
    architecture: "LR + XGBoost ensemble (equal-weight average)",
    trainingData: "17 static + 5 git features from bug-fix commits — Django, Flask, requests, pip, etc.",
    useCase: "Flags high-risk functions before they ship",
    techStack: ["scikit-learn", "XGBoost", "Python"],
    metrics: [
      { label: "LR Test AUC",    value: "0.677", pct: 68 },
      { label: "XGB Test AUC",   value: "0.710", pct: 71 },
      { label: "Ensemble AUC",   value: "0.704", pct: 70 },
      { label: "Train / Test",   value: "2 550 / 450", pct: 85 },
    ],
    description:
      "LR baseline (interpretable) + XGBoost (regularised, early stopping at iter 97) averaged ensemble. Trained on 22 features: 17 static code metrics + 5 git signals. 3 000-sample retrain improved ensemble AUC from 0.641 → 0.704.",
    endpoint: "POST /analyze/bugs",
  },
  {
    id: "clones",
    name: "Code Clone Detection",
    icon: Copy,
    iconColor: "text-blue-400",
    iconBg: "bg-blue-500/10",
    architecture: "TF-IDF vectorization + cosine similarity",
    trainingData: "Tokenized function-level code blocks",
    useCase: "Detects duplicate / near-duplicate code blocks",
    techStack: ["scikit-learn", "Python"],
    metrics: [
      { label: "Similarity Threshold", value: "80%",   pct: 80 },
      { label: "Token Granularity",    value: "Function-level", pct: 85 },
    ],
    description:
      "Splits code into function-level blocks, vectorises with TF-IDF, and identifies clones by cosine similarity ≥ 0.8.",
    endpoint: "POST /analyze/clones",
  },
  {
    id: "refactoring",
    name: "Refactoring Suggester",
    icon: Wrench,
    iconColor: "text-purple-400",
    iconBg: "bg-purple-500/10",
    architecture: "AST visitor pattern + heuristic rules",
    trainingData: "Python AST node analysis",
    useCase: "Long method, god class, feature envy, data clumps",
    techStack: ["ast", "radon", "Python"],
    metrics: [
      { label: "Smell Types Detected", value: "8+",  pct: 80 },
      { label: "Avg Latency",          value: "<10ms", pct: 99 },
    ],
    description:
      "Walks the Python AST to identify refactoring opportunities using Fowler's catalogue: long method, god class, feature envy, etc.",
    endpoint: "POST /analyze/refactoring",
  },
  {
    id: "dead_code",
    name: "Dead Code Detector",
    icon: FileX,
    iconColor: "text-gray-400",
    iconBg: "bg-gray-500/10",
    architecture: "AST visitor + symbol table",
    trainingData: "Definition / reference tracking in Python AST",
    useCase: "Unused variables, functions, classes, imports",
    techStack: ["ast", "Python"],
    metrics: [
      { label: "Dead Code Types", value: "4",     pct: 75 },
      { label: "False Positives", value: "< 5%",  pct: 95 },
    ],
    description:
      "Builds a definition-reference map from the AST to flag symbols that are defined but never used.",
    endpoint: "POST /analyze/dead-code",
  },
  {
    id: "debt",
    name: "Technical Debt Estimator",
    icon: CreditCard,
    iconColor: "text-pink-400",
    iconBg: "bg-pink-500/10",
    architecture: "SQALE-inspired rule engine",
    trainingData: "Code metrics + weighted remediation cost rules",
    useCase: "A–E debt rating, remediation time estimate",
    techStack: ["radon", "astroid", "Python"],
    metrics: [
      { label: "Rating Levels",   value: "A–E",    pct: 100 },
      { label: "Debt Dimensions", value: "5",      pct: 80 },
    ],
    description:
      "SQALE-inspired model combining complexity, duplication, documentation, security and style metrics into a single A–E debt rating with estimated remediation hours.",
    endpoint: "POST /analyze/debt",
  },
  {
    id: "docs",
    name: "Documentation Quality",
    icon: BookOpen,
    iconColor: "text-green-400",
    iconBg: "bg-green-500/10",
    architecture: "Docstring scoring + coverage analysis",
    trainingData: "Docstring presence, format, completeness heuristics",
    useCase: "Docstring coverage, parameter docs, example detection",
    techStack: ["ast", "Python"],
    metrics: [
      { label: "Scoring Dimensions", value: "4",   pct: 80 },
      { label: "Max Score",          value: "100", pct: 100 },
    ],
    description:
      "Scores documentation across 4 dimensions: module-level docs, function docstrings, parameter coverage, and inline examples.",
    endpoint: "POST /analyze/docs",
  },
  {
    id: "performance",
    name: "Performance Analyzer",
    icon: Gauge,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-500/10",
    architecture: "AST pattern matching + complexity analysis",
    trainingData: "Known performance anti-patterns in Python",
    useCase: "N+1 queries, O(n²) loops, string concat, blocking calls",
    techStack: ["ast", "radon", "Python"],
    metrics: [
      { label: "Anti-patterns Detected", value: "10+",  pct: 80 },
      { label: "Severity Levels",        value: "3",    pct: 70 },
    ],
    description:
      "Pattern-matches known performance anti-patterns (nested loops, repeated database calls, string concatenation in loops, blocking I/O) and reports severity.",
    endpoint: "POST /analyze/performance",
  },
  {
    id: "dependencies",
    name: "Dependency Analyzer",
    icon: Network,
    iconColor: "text-indigo-400",
    iconBg: "bg-indigo-500/10",
    architecture: "Fan-out / coupling metrics on import graph",
    trainingData: "Python import AST analysis",
    useCase: "Coupling score, circular dependencies, unstable modules",
    techStack: ["ast", "Python"],
    metrics: [
      { label: "Coupling Metrics", value: "4",    pct: 75 },
      { label: "Dependency Types", value: "3",    pct: 70 },
    ],
    description:
      "Builds an import dependency graph, computes fan-out coupling, detects circular dependencies and flags highly coupled modules.",
    endpoint: "POST /analyze/dependencies",
  },
  {
    id: "readability",
    name: "Readability Scorer",
    icon: Eye,
    iconColor: "text-teal-400",
    iconBg: "bg-teal-500/10",
    architecture: "4-dimension weighted scoring",
    trainingData: "Naming conventions, structure, comments, complexity",
    useCase: "Overall readability score for developers",
    techStack: ["ast", "radon", "Python"],
    metrics: [
      { label: "Scoring Dimensions", value: "4",   pct: 80 },
      { label: "Score Range",        value: "0–100", pct: 100 },
    ],
    description:
      "Combines naming quality, code structure, comment density, and complexity into a single 0–100 readability score.",
    endpoint: "POST /analyze/readability",
  },
];

// ─── Component ────────────────────────────────────────────────────────────────

const TASK_LABELS: Record<string, string> = {
  security: "Security",
  bug: "Bug Prediction",
  complexity: "Complexity",
  pattern: "Pattern",
};

function ALStatusPanel() {
  const [al, setAl] = useState<ALStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    getALStatus()
      .then(setAl)
      .catch(() => setAl(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  if (loading) {
    return (
      <div className="bg-card border border-border rounded-xl p-5 mb-8 animate-pulse h-40" />
    );
  }
  if (!al) return null;

  const latestRound = al.history[0];

  return (
    <div className="bg-card border border-border rounded-xl p-5 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Activity className="w-4 h-4 text-primary" />
          </div>
          <span className="text-sm font-semibold text-foreground">Active Learning Loop</span>
          <span className="text-xs text-muted-foreground ml-1">
            — triggers retrain every {al.trigger_threshold} corrections
          </span>
        </div>
        <button
          onClick={load}
          className="p-1.5 rounded-lg hover:bg-secondary/50 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-3.5 h-3.5 text-muted-foreground" />
        </button>
      </div>

      {/* Per-task progress */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {Object.entries(al.per_task).map(([task, info]) => (
          <div key={task} className="bg-secondary/30 rounded-lg p-3">
            <div className="text-xs font-semibold text-foreground mb-1">
              {TASK_LABELS[task] ?? task}
            </div>
            <div className="h-1.5 bg-secondary rounded-full overflow-hidden mb-1">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary to-primary/60 transition-all"
                style={{ width: `${info.progress_pct}%` }}
              />
            </div>
            <div className="text-[10px] text-muted-foreground">
              {info.feedback_since_last_trigger} / {al.trigger_threshold} corrections
            </div>
          </div>
        ))}
      </div>

      {/* Total + last round */}
      <div className="flex items-center gap-6 text-xs text-muted-foreground flex-wrap">
        <span>Total feedback: <strong className="text-foreground">{al.total_feedback}</strong></span>
        <span>Retrain rounds: <strong className="text-foreground">{al.history.length}</strong></span>
        {latestRound && (
          <>
            <span>
              Last run:{" "}
              <strong className="text-foreground">
                {new Date(latestRound.triggered_at).toLocaleString()}
              </strong>
            </span>
            <span className={`font-medium ${latestRound.deployed ? "text-green-400" : "text-muted-foreground"}`}>
              {latestRound.deployed ? "Deployed" : `Skipped — ${latestRound.reason ?? "below quality gate"}`}
            </span>
            {latestRound.pre_auc !== null && latestRound.post_auc !== null && (
              <span>
                AUC: {latestRound.pre_auc.toFixed(3)} &rarr;{" "}
                <strong className={latestRound.post_auc > latestRound.pre_auc ? "text-green-400" : "text-red-400"}>
                  {latestRound.post_auc.toFixed(3)}
                </strong>
              </span>
            )}
            {latestRound.pre_ece !== null && latestRound.post_ece !== null && (
              <span>
                ECE: {latestRound.pre_ece.toFixed(4)} &rarr;{" "}
                <strong className={latestRound.post_ece < latestRound.pre_ece ? "text-green-400" : "text-red-400"}>
                  {latestRound.post_ece.toFixed(4)}
                </strong>
              </span>
            )}
            {latestRound.hard_negatives_mined !== undefined && latestRound.hard_negatives_mined > 0 && (
              <span className="text-blue-400">
                +{latestRound.hard_negatives_mined} hard negatives mined
              </span>
            )}
          </>
        )}
        {al.history.length === 0 && (
          <span className="italic">No retrains yet — submit corrections to start the loop.</span>
        )}
      </div>
    </div>
  );
}

function LOPOTable({ data, metric }: { data: LOPOResult; metric: "auc" | "spearman" }) {
  if (!data?.project_results?.length) return null;
  const rows = data.project_results.filter(r => r[metric] !== null && r[metric] !== undefined);
  if (!rows.length) return null;
  const mean = metric === "auc" ? data.mean_auc : data.mean_spearman;
  const std  = metric === "auc" ? data.std_auc  : data.std_spearman;
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          {metric === "auc" ? "AUC" : "Spearman"} per held-out repo
        </span>
        {mean !== null && mean !== undefined && (
          <span className="text-xs text-foreground font-mono">
            mean={mean.toFixed(3)} ±{std?.toFixed(3) ?? "–"}
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const val = r[metric] as number;
          const pct = Math.round(Math.abs(val) * 100);
          const isWeak = metric === "auc" ? val < 0.55 : val < 0.6;
          return (
            <div key={r.held_out_repo} className="flex items-center gap-2">
              <span className="text-[11px] text-muted-foreground w-40 truncate shrink-0" title={r.held_out_repo}>
                {r.held_out_repo.split("/").pop()}
              </span>
              <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${isWeak ? "bg-yellow-500/70" : "bg-green-500/70"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`text-[11px] font-mono w-12 text-right ${isWeak ? "text-yellow-400" : "text-green-400"}`}>
                {val.toFixed(3)}
              </span>
              <span className="text-[10px] text-muted-foreground/50">n={r.n_test}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LOPOPanel() {
  const [eval_, setEval] = useState<ModelsEvaluation | null>(null);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    getModelsEvaluation()
      .then(setEval)
      .catch(() => setEval(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="bg-card border border-border rounded-xl p-5 mb-8 animate-pulse h-24" />
  );
  if (!eval_) return null;

  const tb = eval_.temporal_bug;

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden mb-8">
      <button
        className="w-full flex items-center justify-between p-5 hover:bg-secondary/20 transition-colors"
        onClick={() => setOpen(v => !v)}
      >
        <div className="flex items-center gap-2">
          <div className="p-2 bg-yellow-500/10 rounded-lg">
            <TrendingUp className="w-4 h-4 text-yellow-400" />
          </div>
          <span className="text-sm font-semibold text-foreground">Cross-Project Generalisation (LOPO)</span>
          <span className="text-xs text-muted-foreground ml-1">— leave-one-project-out evaluation</span>
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
      </button>

      {open && (
        <div className="px-5 pb-5 space-y-6 border-t border-border pt-4">
          {/* Temporal split callout */}
          {tb?.delta_auc !== null && tb?.delta_auc !== undefined && (
            <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/8 p-3 text-xs space-y-1">
              <div className="flex items-center gap-2 font-semibold text-yellow-400">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                Temporal evaluation — bug prediction
              </div>
              <p className="text-muted-foreground">
                Random split AUC: <span className="text-foreground font-mono">{tb.random_split_auc?.toFixed(3)}</span>
                {" "}&rarr; Temporal split AUC: <span className={`font-mono ${(tb.temporal_split_auc ?? 0) < 0.55 ? "text-yellow-400" : "text-foreground"}`}>{tb.temporal_split_auc?.toFixed(3)}</span>
                {" "}(delta: <span className="text-red-400 font-mono">{tb.delta_auc.toFixed(3)}</span>).
                {" "}Trains on older commits, tests on newer — prevents SZZ-style label leakage.
              </p>
            </div>
          )}

          {/* Security LOPO */}
          {eval_.security?.project_results && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Shield className="w-3.5 h-3.5 text-red-400 shrink-0" />
                <span className="text-xs font-semibold text-foreground">Security — {eval_.security.n_projects} repos</span>
                <span className="text-[10px] text-muted-foreground italic ml-1">
                  (mean AUC {eval_.security.mean_auc?.toFixed(3) ?? "–"} — near chance; contrastive pre-training targets 0.65+)
                </span>
              </div>
              <LOPOTable data={eval_.security} metric="auc" />
            </div>
          )}

          {/* Bug LOPO */}
          {eval_.bug?.project_results && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Bug className="w-3.5 h-3.5 text-orange-400 shrink-0" />
                <span className="text-xs font-semibold text-foreground">Bug Prediction — {eval_.bug.n_projects} repos</span>
              </div>
              <LOPOTable data={eval_.bug} metric="auc" />
            </div>
          )}

          {/* Complexity LOPO */}
          {eval_.complexity?.project_results && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <BarChart2 className="w-3.5 h-3.5 text-yellow-400 shrink-0" />
                <span className="text-xs font-semibold text-foreground">Complexity — {eval_.complexity.n_projects} repos</span>
              </div>
              <LOPOTable data={eval_.complexity} metric="spearman" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const Models = () => {
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = (id: string) => setExpanded((prev) => (prev === id ? null : id));

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Cpu className="w-5 h-5 text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">ML Models</h1>
        </div>
        <p className="text-muted-foreground mb-8">
          12 specialised models trained on synthetic + open-source code — all run in parallel on every analysis.
        </p>

        {/* Summary bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          {[
            { icon: CheckCircle2, color: "text-green-400", label: "Production Models", value: "12" },
            { icon: Database,     color: "text-primary",   label: "Training Samples",  value: "4 500+" },
            { icon: TrendingUp,   color: "text-orange-400",label: "Avg AUC / R²",      value: "0.82" },
            { icon: Layers,       color: "text-yellow-400",label: "Endpoints",          value: "13" },
          ].map(({ icon: Icon, color, label, value }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-4 flex items-center gap-3">
              <div className="p-2 bg-secondary rounded-lg shrink-0">
                <Icon className={`w-4 h-4 ${color}`} />
              </div>
              <div>
                <div className="text-xl font-bold text-foreground">{value}</div>
                <div className="text-xs text-muted-foreground">{label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Cross-project generalisation panel */}
        <LOPOPanel />

        {/* Active learning loop status */}
        <ALStatusPanel />

        {/* Model list */}
        <div className="space-y-3">
          {ALL_MODELS.map((model, idx) => {
            const Icon = model.icon;
            const isOpen = expanded === model.id;
            return (
              <div key={model.id} className="bg-card border border-border rounded-xl overflow-hidden">
                {/* Row */}
                <div
                  className="flex items-center gap-4 p-5 cursor-pointer hover:bg-secondary/20 transition-colors"
                  onClick={() => toggle(model.id)}
                >
                  <div className="text-xs font-bold text-muted-foreground w-5 shrink-0">{idx + 1}</div>
                  <div className={`p-2.5 ${model.iconBg} rounded-xl shrink-0`}>
                    <Icon className={`w-5 h-5 ${model.iconColor}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-0.5">
                      <span className="font-semibold text-foreground">{model.name}</span>
                      <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-[10px]">
                        Production
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground truncate">{model.useCase}</p>
                  </div>
                  {/* top 2 metrics inline */}
                  <div className="hidden md:flex gap-5 shrink-0">
                    {model.metrics.slice(0, 2).map((m) => (
                      <div key={m.label} className="text-center">
                        <div className="text-sm font-bold text-foreground">{m.value}</div>
                        <div className="text-[10px] text-muted-foreground whitespace-nowrap">{m.label}</div>
                      </div>
                    ))}
                  </div>
                  <div className="shrink-0">
                    {isOpen ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
                  </div>
                </div>

                {/* Expanded */}
                {isOpen && (
                  <div className="border-t border-border px-5 pb-5 pt-4">
                    <p className="text-sm text-muted-foreground mb-5">{model.description}</p>
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Left: arch + data */}
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            <Layers className="w-3.5 h-3.5" /> Architecture
                          </div>
                          <p className="text-sm text-foreground bg-secondary/30 rounded-lg px-3 py-2 border border-border">
                            {model.architecture}
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            <Database className="w-3.5 h-3.5" /> Training Data
                          </div>
                          <p className="text-sm text-foreground bg-secondary/30 rounded-lg px-3 py-2 border border-border">
                            {model.trainingData}
                          </p>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            <Network className="w-3.5 h-3.5" /> API Endpoint
                          </div>
                          <code className="text-sm text-primary bg-secondary/30 rounded-lg px-3 py-2 border border-border block font-mono">
                            {model.endpoint}
                          </code>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {model.techStack.map((t) => (
                            <span key={t} className="px-2 py-0.5 bg-secondary text-xs text-muted-foreground rounded-md border border-border">
                              {t}
                            </span>
                          ))}
                        </div>
                      </div>
                      {/* Right: metrics bars */}
                      <div>
                        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                          Performance Metrics
                        </div>
                        <div className="space-y-3">
                          {model.metrics.map((m) => (
                            <div key={m.label}>
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-sm text-foreground">{m.label}</span>
                                <span className="text-sm font-semibold text-foreground">{m.value}</span>
                              </div>
                              <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-primary to-primary/60"
                                  style={{ width: `${m.pct}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </main>
    </div>
  );
};

export default Models;
