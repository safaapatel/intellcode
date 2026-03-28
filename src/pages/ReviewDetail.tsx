import { useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle,
  Shield,
  Zap,
  Bug,
  Copy,
  FileX,
  Wrench,
  CreditCard,
  BookOpen,
  Gauge,
  Network,
  Eye,
  AlignLeft,
  Download,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Flag,
  X,
  Brain,
  Lightbulb,
  ClipboardCopy,
  Check,
} from "lucide-react";
import { mockReviewResult, mockIssues } from "@/data/mockData";
import { getEntry } from "@/services/reviewHistory";
import { getSession } from "@/services/auth";
import { toast } from "sonner";
import type { FullAnalysisResult } from "@/types/analysis";
import { submitAnalysisFeedback } from "@/services/api";

// ─── Feedback types ────────────────────────────────────────────────────────────

interface FindingFeedback {
  status: "accepted" | "rejected" | "false_positive";
  reasoning: string;
  reviewer: string;
  at: string;
}

const FEEDBACK_KEY = "intellcode_feedback";

function loadFeedback(filename: string): Record<string, FindingFeedback> {
  try {
    const raw = localStorage.getItem(FEEDBACK_KEY);
    const all = raw ? JSON.parse(raw) : {};
    return all[filename] ?? {};
  } catch {
    return {};
  }
}

function persistFeedback(filename: string, feedbacks: Record<string, FindingFeedback>) {
  try {
    const raw = localStorage.getItem(FEEDBACK_KEY);
    const all = raw ? JSON.parse(raw) : {};
    all[filename] = feedbacks;
    localStorage.setItem(FEEDBACK_KEY, JSON.stringify(all));
  } catch {
    // ignore
  }
}

const FB_BADGE: Record<FindingFeedback["status"], string> = {
  accepted: "bg-green-500/20 text-green-400 border border-green-500/30",
  rejected: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
  false_positive: "bg-muted text-muted-foreground border border-border",
};

const FB_LABEL: Record<FindingFeedback["status"], string> = {
  accepted: "Accepted",
  rejected: "Rejected",
  false_positive: "False Positive",
};

// ─── Analysis-level feedback bar ──────────────────────────────────────────────

function AnalysisFeedbackBar({ result }: { result: FullAnalysisResult }) {
  const [sent, setSent] = useState(false);
  const [selected, setSelected] = useState<"positive" | "negative" | null>(null);
  const [comment, setComment] = useState("");
  const [showComment, setShowComment] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (rating: "positive" | "negative") => {
    setSelected(rating);
    setSubmitting(true);
    try {
      await submitAnalysisFeedback({
        filename: result.filename,
        overall_score: result.overall_score ?? 0,
        rating,
        comment: comment.trim() || undefined,
      });
    } catch { /* backend offline — ignore */ }
    setSent(true);
    setSubmitting(false);
    toast.success(rating === "positive" ? "Thanks! Glad it was helpful." : "Thanks for the feedback — we'll use it to improve.");
  };

  if (sent) {
    return (
      <div className="mt-6 rounded-xl border border-border bg-card px-6 py-4 flex items-center gap-3 text-sm text-muted-foreground">
        {selected === "positive"
          ? <ThumbsUp className="w-4 h-4 text-green-500" />
          : <ThumbsDown className="w-4 h-4 text-orange-400" />}
        Feedback received — thank you!
      </div>
    );
  }

  return (
    <div className="mt-6 rounded-xl border border-border bg-card px-6 py-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <span className="text-sm text-muted-foreground">Was this analysis accurate and helpful?</span>
        <div className="flex items-center gap-2">
          {showComment && (
            <input
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Optional comment…"
              className="text-sm bg-input border border-border rounded-lg px-3 py-1.5 text-foreground w-56 focus:outline-none focus:ring-1 focus:ring-primary"
            />
          )}
          <button
            onClick={() => setShowComment((v) => !v)}
            className="text-xs text-muted-foreground hover:text-foreground px-2 py-1.5 rounded-lg border border-border hover:border-muted-foreground transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5 inline mr-1" />
            Comment
          </button>
          <button
            disabled={submitting}
            onClick={() => submit("positive")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border hover:border-green-500 hover:bg-green-500/10 hover:text-green-500 text-muted-foreground text-sm transition-colors disabled:opacity-50"
          >
            <ThumbsUp className="w-4 h-4" /> Yes
          </button>
          <button
            disabled={submitting}
            onClick={() => submit("negative")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border hover:border-orange-400 hover:bg-orange-400/10 hover:text-orange-400 text-muted-foreground text-sm transition-colors disabled:opacity-50"
          >
            <ThumbsDown className="w-4 h-4" /> No
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const scoreStroke = (s: number) =>
  s >= 90 ? "stroke-green-500" : s >= 75 ? "stroke-primary" : s >= 60 ? "stroke-yellow-500" : "stroke-destructive";

const scoreText = (s: number) =>
  s >= 90 ? "text-green-500" : s >= 75 ? "text-primary" : s >= 60 ? "text-yellow-500" : "text-destructive";

const RATING_CLS: Record<string, string> = {
  A: "text-green-500 bg-green-500/10 border-green-500/30",
  B: "text-primary bg-primary/10 border-primary/30",
  C: "text-yellow-500 bg-yellow-500/10 border-yellow-500/30",
  D: "text-orange-500 bg-orange-500/10 border-orange-500/30",
  E: "text-destructive bg-destructive/10 border-destructive/30",
  F: "text-destructive bg-destructive/10 border-destructive/30",
};

const SEV_CLS: Record<string, string> = {
  critical: "bg-destructive text-destructive-foreground",
  high: "bg-orange-500 text-white",
  medium: "bg-yellow-500 text-black",
  low: "bg-primary text-primary-foreground",
  warning: "bg-orange-500 text-white",
  info: "bg-muted text-muted-foreground",
};

const SEV_BORDER: Record<string, string> = {
  critical: "border-l-destructive",
  high:     "border-l-orange-500",
  medium:   "border-l-yellow-500",
  low:      "border-l-primary",
  warning:  "border-l-orange-500",
  info:     "border-l-border",
};

function ScoreCircle({ score, label, size = 128, danger = false }: { score: number; label: string; size?: number; danger?: boolean }) {
  const r = size / 2 - 8;
  const circ = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, score));
  // danger=true inverts colour logic: high value = bad = red (used for bug probability)
  const colorScore = danger ? 100 - clamped : clamped;
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="w-full h-full -rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none"
            className={scoreStroke(colorScore)} strokeWidth="8"
            strokeDasharray={`${(clamped / 100) * circ} ${circ}`} strokeLinecap="round" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${size >= 100 ? "text-3xl" : "text-xl"} ${scoreText(colorScore)}`}>
            {Math.round(clamped)}
          </span>
          <span className="text-[10px] text-muted-foreground">/100</span>
        </div>
      </div>
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

function Badge({ label, cls }: { label: string; cls?: string }) {
  const base = cls ?? "bg-muted text-muted-foreground";
  return <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${base}`}>{label}</span>;
}

function SevBadge({ sev }: { sev: string }) {
  return <Badge label={sev.toUpperCase()} cls={SEV_CLS[sev] ?? SEV_CLS.info} />;
}

function RatingBadge({ rating }: { rating: string }) {
  const cls = RATING_CLS[rating] ?? RATING_CLS.C;
  return <span className={`inline-block px-3 py-1 rounded-full text-sm font-bold border ${cls}`}>{rating}</span>;
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="flex flex-col items-center py-10 text-muted-foreground gap-2">
      <CheckCircle className="w-10 h-10 text-green-500/50" />
      <p className="text-sm">{msg}</p>
    </div>
  );
}

// ─── Tab bodies ───────────────────────────────────────────────────────────────

function OverviewTab({ r }: { r: FullAnalysisResult }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-end justify-center py-2">
        <ScoreCircle score={r.overall_score} label="Overall" />
        {r.complexity && <ScoreCircle score={r.complexity.score} label="Complexity" size={96} />}
        {r.readability && <ScoreCircle score={r.readability.overall_score} label="Readability" size={96} />}
        {r.docs && <ScoreCircle score={r.docs.average_quality} label="Docs" size={96} />}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          {
            label: "Security Issues",
            value: r.security?.summary?.total ?? 0,
            sub: `${r.security?.summary?.critical ?? 0} critical`,
            cls: (r.security?.summary?.critical ?? 0) > 0 ? "text-destructive" : "text-foreground",
            border: (r.security?.summary?.critical ?? 0) > 0 ? "border-l-destructive" : (r.security?.summary?.total ?? 0) > 0 ? "border-l-orange-500" : "border-l-green-500",
          },
          {
            label: "Bug Risk",
            value: (r.bug_prediction?.risk_level ?? "unknown").toUpperCase(),
            sub: `${Math.round((r.bug_prediction?.bug_probability ?? 0) * 100)}% probability`,
            cls: ({ low: "text-green-500", medium: "text-yellow-500", high: "text-orange-500", critical: "text-destructive" } as Record<string, string>)[r.bug_prediction?.risk_level ?? ""] ?? "text-foreground",
            border: ({ low: "border-l-green-500", medium: "border-l-yellow-500", high: "border-l-orange-500", critical: "border-l-destructive" } as Record<string, string>)[r.bug_prediction?.risk_level ?? ""] ?? "border-l-primary",
          },
          {
            label: "Clones Found",
            value: r.clones?.clones?.length ?? 0,
            sub: `${((r.clones?.clone_rate ?? 0) * 100).toFixed(0)}% duplication`,
            cls: "text-foreground",
            border: (r.clones?.clones?.length ?? 0) > 0 ? "border-l-yellow-500" : "border-l-green-500",
          },
          {
            label: "Dead Code",
            value: `${r.dead_code?.dead_line_count ?? 0} lines`,
            sub: `${((r.dead_code?.dead_ratio ?? 0) * 100).toFixed(1)}% of file`,
            cls: "text-foreground",
            border: (r.dead_code?.dead_line_count ?? 0) > 0 ? "border-l-primary" : "border-l-green-500",
          },
        ].map((card) => (
          <div key={card.label} className={`bg-secondary/40 rounded-xl p-4 text-center border-l-4 ${card.border}`}>
            <p className={`text-2xl font-bold ${card.cls}`}>{card.value}</p>
            <p className="text-xs text-muted-foreground font-medium mt-1">{card.label}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{card.sub}</p>
          </div>
        ))}
      </div>

      <div className="bg-secondary/20 rounded-xl p-4 text-sm text-muted-foreground leading-relaxed">
        {r.summary}
      </div>

      {r.status === "critical" && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-destructive" />
          <span className="text-destructive font-medium">Critical issues found — immediate action required</span>
        </div>
      )}
      {r.status === "action_required" && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-500" />
          <span className="text-yellow-500 font-medium">Review complete — action required</span>
        </div>
      )}
      {r.status === "clean" && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-green-500" />
          <span className="text-green-500 font-medium">Code looks clean — no major issues detected</span>
        </div>
      )}
    </div>
  );
}

function SecurityTab({
  r,
  feedbacks,
  onReview,
}: {
  r: FullAnalysisResult;
  feedbacks: Record<string, FindingFeedback>;
  onReview: (key: string, title: string) => void;
}) {
  const { findings, summary } = r.security;
  if (findings.length === 0) return <Empty msg="No security vulnerabilities detected." />;
  return (
    <div className="space-y-4">
      <div className="flex gap-3 text-sm flex-wrap mb-2">
        {(["critical", "high", "medium", "low"] as const).map((sev) => (
          <span key={sev} className="flex items-center gap-1">
            <SevBadge sev={sev} />
            <span className="text-muted-foreground">{summary[sev]}</span>
          </span>
        ))}
      </div>
      {findings.map((f, i) => {
        const key = `${f.title}-${f.lineno}`;
        const fb = feedbacks[key];
        return (
          <div key={i} className={`bg-secondary/30 rounded-xl p-4 space-y-2 border-l-4 ${SEV_BORDER[f.severity] ?? "border-l-border"}`}>
            <div className="flex items-center gap-2 flex-wrap">
              <SevBadge sev={f.severity} />
              <span className="font-semibold text-foreground text-sm">{f.title}</span>
              <span className="text-xs text-muted-foreground ml-auto">{f.cwe} · line {f.lineno}</span>
            </div>
            <p className="text-sm text-muted-foreground">{f.description}</p>
            {f.snippet && (
              <pre className="bg-destructive/10 border border-destructive/20 rounded p-3 text-xs font-mono text-foreground overflow-x-auto">
                {f.snippet}
              </pre>
            )}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <p className="text-xs text-muted-foreground">
                Source: {f.source} · Confidence: {Math.round(f.confidence * 100)}%
              </p>
              {fb ? (
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${FB_BADGE[fb.status]}`}>
                    {FB_LABEL[fb.status]}
                  </span>
                  <span className="text-xs text-muted-foreground">by {fb.reviewer}</span>
                  <button
                    className="text-xs text-primary hover:underline"
                    onClick={() => onReview(key, f.title)}
                  >
                    Edit
                  </button>
                </div>
              ) : (
                <button
                  className="flex items-center gap-1 text-xs text-primary hover:underline"
                  onClick={() => onReview(key, f.title)}
                >
                  <MessageSquare className="w-3 h-3" />
                  Add Review
                </button>
              )}
            </div>
            {fb?.reasoning && (
              <p className="text-xs text-muted-foreground italic border-l-2 border-border pl-2">
                "{fb.reasoning}"
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ComplexityTab({ r }: { r: FullAnalysisResult }) {
  const c = r.complexity;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <ScoreCircle score={c.score} label={`Grade ${c.grade}`} />
        <div className="grid grid-cols-2 gap-3 flex-1">
          {[
            { label: "Cyclomatic CC", value: c.cyclomatic },
            { label: "Cognitive CC", value: c.cognitive },
            { label: "Halstead Bugs", value: c.halstead_bugs.toFixed(2) },
            { label: "Maintainability", value: c.maintainability_index.toFixed(0) },
            { label: "Source Lines", value: c.sloc },
            { label: "Lines > 80 chars", value: c.n_lines_over_80 },
            { label: "Long Functions", value: c.n_long_functions },
            { label: "Complex Functions", value: c.n_complex_functions },
          ].map((m) => (
            <div key={m.label} className="bg-secondary/30 rounded-lg p-3">
              <p className="text-lg font-bold text-foreground">{m.value}</p>
              <p className="text-xs text-muted-foreground">{m.label}</p>
            </div>
          ))}
        </div>
      </div>
      {c.function_issues.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">Function Issues</h3>
          <div className="space-y-2">
            {c.function_issues.map((fi, i) => (
              <div key={i} className="bg-secondary/20 rounded-lg p-3 flex items-center gap-3 text-sm">
                <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0" />
                <span className="font-mono text-foreground">{fi.name}</span>
                <span className="text-muted-foreground">{fi.issue} ({fi.value})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function BugTab({ r }: { r: FullAnalysisResult }) {
  const b = r.bug_prediction;
  const pct = Math.round(b.bug_probability * 100);
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <ScoreCircle score={pct} label="Bug Probability %" danger />
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-sm">Risk level:</span>
            <SevBadge sev={b.risk_level} />
          </div>
          <p className="text-sm text-muted-foreground">Model confidence: {Math.round(b.confidence * 100)}%</p>
          <p className="text-sm text-muted-foreground">Static score: {Math.round(b.static_score * 100)}%</p>
          {b.git_score !== null && (
            <p className="text-sm text-muted-foreground">Git score: {Math.round((b.git_score ?? 0) * 100)}%</p>
          )}
        </div>
      </div>
      {b.risk_factors.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">Risk Factors</h3>
          <ul className="space-y-2">
            {b.risk_factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ClonesTab({ r }: { r: FullAnalysisResult }) {
  const c = r.clones;
  if (c.clones.length === 0) return <Empty msg="No code clones detected." />;
  const typeLabel: Record<string, string> = { type1: "Exact", type2: "Renamed", type3: "Near-Miss" };
  const typeSev: Record<string, string> = { type1: "critical", type2: "high", type3: "medium" };
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-2">
        <span>Duplication score: <strong className="text-foreground">{c.duplication_score.toFixed(0)}/100</strong></span>
        <span>Blocks: <strong className="text-foreground">{c.total_blocks}</strong></span>
      </div>
      {c.clones.map((clone, i) => (
        <div key={i} className="bg-secondary/30 rounded-xl p-4 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <SevBadge sev={typeSev[clone.clone_type] ?? "info"} />
            <Badge label={typeLabel[clone.clone_type] ?? clone.clone_type} />
            <span className="text-sm text-muted-foreground ml-auto">
              Similarity: {(clone.similarity * 100).toFixed(0)}%
            </span>
          </div>
          <div className="font-mono text-sm">
            <span className="text-primary">{clone.block_a}</span>
            <span className="text-muted-foreground"> ↔ </span>
            <span className="text-primary">{clone.block_b}</span>
          </div>
          <p className="text-xs text-muted-foreground">{clone.description}</p>
          <p className="text-xs text-muted-foreground">Lines: {clone.start_line_a} & {clone.start_line_b}</p>
        </div>
      ))}
    </div>
  );
}

function RefactoringTab({ r }: { r: FullAnalysisResult }) {
  const rf = r.refactoring;
  if (rf.suggestions.length === 0) return <Empty msg="No refactoring opportunities found." />;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-2">
        <span>Total effort: <strong className="text-foreground">{rf.total_effort_minutes} min</strong></span>
        {Object.entries(rf.priority_counts).map(([p, n]) => (
          <span key={p}><SevBadge sev={p} /> {n}</span>
        ))}
      </div>
      {rf.suggestions.map((s, i) => (
        <div key={i} className={`bg-secondary/30 rounded-xl p-4 space-y-2 border-l-4 ${SEV_BORDER[s.priority] ?? "border-l-border"}`}>
          <div className="flex items-center gap-2 flex-wrap">
            <SevBadge sev={s.priority} />
            <span className="font-semibold text-foreground text-sm">{s.title}</span>
            <span className="text-xs text-muted-foreground ml-auto">{s.effort_minutes} min</span>
          </div>
          <p className="text-sm text-muted-foreground">{s.description}</p>
          <p className="text-xs font-mono text-muted-foreground">{s.location}</p>
        </div>
      ))}
    </div>
  );
}

function DeadCodeTab({ r }: { r: FullAnalysisResult }) {
  const d = r.dead_code;
  if (d.issues.length === 0) return <Empty msg="No dead code detected." />;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-2">
        <span>Dead lines: <strong className="text-foreground">{d.dead_line_count}</strong></span>
        <span>Dead ratio: <strong className="text-foreground">{(d.dead_ratio * 100).toFixed(1)}%</strong></span>
      </div>
      {d.issues.map((issue, i) => (
        <div key={i} className={`bg-secondary/30 rounded-xl p-4 space-y-1 border-l-4 ${SEV_BORDER[issue.severity] ?? "border-l-border"}`}>
          <div className="flex items-center gap-2 flex-wrap">
            <SevBadge sev={issue.severity} />
            {issue.removable && (
              <span className="text-xs text-green-500 font-medium">removable</span>
            )}
            <span className="font-semibold text-sm text-foreground">{issue.title}</span>
          </div>
          <p className="text-sm text-muted-foreground">{issue.description}</p>
          <p className="text-xs font-mono text-muted-foreground">{issue.location}</p>
        </div>
      ))}
    </div>
  );
}

function DebtTab({ r }: { r: FullAnalysisResult }) {
  const d = r.technical_debt;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <div className="text-center space-y-1">
          <RatingBadge rating={d.overall_rating} />
          <p className="text-xs text-muted-foreground">Overall Rating</p>
        </div>
        <div className="grid grid-cols-3 gap-3 flex-1">
          <div className="bg-secondary/30 rounded-lg p-3">
            <p className="text-lg font-bold text-foreground">{d.total_debt_minutes} min</p>
            <p className="text-xs text-muted-foreground">Total Debt</p>
          </div>
          <div className="bg-secondary/30 rounded-lg p-3">
            <p className="text-lg font-bold text-foreground">{d.payoff_days.toFixed(1)} days</p>
            <p className="text-xs text-muted-foreground">Payoff Time</p>
          </div>
          <div className="bg-secondary/30 rounded-lg p-3">
            <p className="text-lg font-bold text-foreground">{d.interest_per_day.toFixed(1)} min/day</p>
            <p className="text-xs text-muted-foreground">Interest</p>
          </div>
        </div>
      </div>
      <div className="space-y-3">
        {d.categories.map((cat) => (
          <div key={cat.name} className="bg-secondary/20 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-foreground capitalize">{cat.name.replace("_", " ")}</span>
              <div className="flex items-center gap-2">
                <RatingBadge rating={cat.rating} />
                <span className="text-sm text-muted-foreground">{cat.debt_minutes} min</span>
              </div>
            </div>
            <ul className="space-y-0.5">
              {cat.items.map((item, i) => (
                <li key={i} className="text-xs text-muted-foreground">· {item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function DocsTab({ r }: { r: FullAnalysisResult }) {
  const d = r.docs;
  if (!d) return <Empty msg="Documentation analysis not available." />;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <ScoreCircle score={d.average_quality} label={`Grade ${d.grade}`} />
        <div className="grid grid-cols-2 gap-3 flex-1">
          <div className="bg-secondary/30 rounded-lg p-3">
            <p className="text-lg font-bold text-foreground">{(d.coverage * 100).toFixed(0)}%</p>
            <p className="text-xs text-muted-foreground">Coverage</p>
          </div>
          <div className="bg-secondary/30 rounded-lg p-3">
            <p className="text-lg font-bold text-foreground">{d.documented_symbols}/{d.total_symbols}</p>
            <p className="text-xs text-muted-foreground">Documented</p>
          </div>
        </div>
      </div>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-muted-foreground border-b border-border">
              <th className="pb-2 pr-4">Symbol</th>
              <th className="pb-2 pr-4">Kind</th>
              <th className="pb-2 pr-4">Score</th>
              <th className="pb-2">Issues</th>
            </tr>
          </thead>
          <tbody>
            {d.symbol_scores.slice(0, 15).map((s, i) => (
              <tr key={i} className="border-b border-border/40">
                <td className="py-2 pr-4 font-mono text-foreground">{s.name}</td>
                <td className="py-2 pr-4 text-muted-foreground">{s.kind}</td>
                <td className={`py-2 pr-4 font-bold ${s.has_docstring ? scoreText(s.quality_score) : "text-destructive"}`}>
                  {s.has_docstring ? Math.round(s.quality_score) : "none"}
                </td>
                <td className="py-2 text-muted-foreground text-xs">{s.issues.join(", ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerformanceTab({ r }: { r: FullAnalysisResult }) {
  const p = r.performance;
  if (!p) return <Empty msg="Performance analysis not available." />;
  if (p.issues.length === 0) return <Empty msg="No performance hotspots detected." />;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-2">
        <span>Hotspot score: <strong className="text-foreground">{p.hotspot_score.toFixed(0)}/100</strong></span>
        {Object.entries(p.severity_counts).map(([sev, n]) => (
          <span key={sev}><SevBadge sev={sev} /> {n}</span>
        ))}
      </div>
      {p.issues.map((issue, i) => (
        <div key={i} className={`bg-secondary/30 rounded-xl p-4 space-y-2 border-l-4 ${SEV_BORDER[issue.severity] ?? "border-l-border"}`}>
          <div className="flex items-center gap-2 flex-wrap">
            <SevBadge sev={issue.severity} />
            <span className="font-semibold text-sm text-foreground">{issue.title}</span>
          </div>
          <p className="text-sm text-muted-foreground">{issue.description}</p>
          <div className="flex items-center gap-1.5 text-xs text-primary">
            <Zap className="w-3 h-3" />
            {issue.speedup_hint}
          </div>
          <p className="text-xs font-mono text-muted-foreground">{issue.location}</p>
        </div>
      ))}
    </div>
  );
}

function DepsTab({ r }: { r: FullAnalysisResult }) {
  const d = r.dependencies;
  if (!d) return <Empty msg="Dependency analysis not available." />;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Total Fan-Out", value: d.fan_out },
          { label: "External", value: d.fan_out_external },
          { label: "Stdlib", value: d.fan_out_stdlib },
          { label: "Local", value: d.fan_out_local },
        ].map((m) => (
          <div key={m.label} className="bg-secondary/30 rounded-lg p-3 text-center">
            <p className="text-xl font-bold text-foreground">{m.value}</p>
            <p className="text-xs text-muted-foreground">{m.label}</p>
          </div>
        ))}
      </div>
      <p className="text-sm text-muted-foreground">
        Instability: <strong className="text-foreground">{(d.instability * 100).toFixed(0)}%</strong>
        &nbsp;·&nbsp;Coupling score: <strong className="text-foreground">{d.coupling_score.toFixed(0)}/100</strong>
      </p>
      {Object.entries(d.dependency_map).map(([cat, mods]) =>
        mods.length > 0 ? (
          <div key={cat}>
            <h3 className="text-sm font-semibold text-foreground mb-2 capitalize">{cat} dependencies</h3>
            <div className="flex flex-wrap gap-2">
              {mods.map((m) => (
                <span key={m} className="font-mono text-xs bg-secondary px-2 py-1 rounded text-foreground">{m}</span>
              ))}
            </div>
          </div>
        ) : null
      )}
      {d.issues.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">Coupling Issues</h3>
          {d.issues.map((issue, i) => (
            <div key={i} className="bg-secondary/20 rounded-lg p-3 mb-2 space-y-1">
              <div className="flex items-center gap-2">
                <SevBadge sev={issue.severity} />
                <span className="text-sm font-medium text-foreground">{issue.title}</span>
              </div>
              <p className="text-xs text-muted-foreground">{issue.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReadabilityTab({ r }: { r: FullAnalysisResult }) {
  const rd = r.readability;
  if (!rd) return <Empty msg="Readability analysis not available." />;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6">
        <ScoreCircle score={rd.overall_score} label={`Grade ${rd.grade}`} />
        <div className="space-y-3 flex-1">
          {rd.dimensions.map((dim) => (
            <div key={dim.name}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-foreground capitalize">{dim.name.replace(/_/g, " ")}</span>
                <span className={`text-sm font-bold ${scoreText(dim.score)}`}>{Math.round(dim.score)}</span>
              </div>
              <div className="w-full bg-secondary rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${dim.score >= 75 ? "bg-primary" : dim.score >= 50 ? "bg-yellow-500" : "bg-destructive"}`}
                  style={{ width: `${dim.score}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
      {rd.top_improvements.length > 0 && (
        <div className="bg-primary/5 border border-primary/20 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-foreground mb-2">Top Improvements</h3>
          <ul className="space-y-2">
            {rd.top_improvements.map((tip, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                <CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function PatternTab({ r }: { r: FullAnalysisResult }) {
  const p = r.patterns;
  if (!p) return <Empty msg="Pattern analysis not available." />;

  const LABEL_COLORS: Record<string, string> = {
    clean: "text-green-500",
    code_smell: "text-yellow-500",
    anti_pattern: "text-orange-500",
    style_violation: "text-primary",
  };
  const LABEL_BG: Record<string, string> = {
    clean: "bg-green-500",
    code_smell: "bg-yellow-500",
    anti_pattern: "bg-orange-500",
    style_violation: "bg-primary",
  };
  const LABEL_DESC: Record<string, string> = {
    clean: "No significant code quality issues detected.",
    code_smell: "Code has structural issues that may hinder maintainability.",
    anti_pattern: "Common anti-patterns found that should be refactored.",
    style_violation: "Code style deviates from best practices.",
  };

  const labelDisplay = p.label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className={`p-4 rounded-full bg-secondary/40`}>
          <Brain className={`w-8 h-8 ${LABEL_COLORS[p.label] ?? "text-foreground"}`} />
        </div>
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Detected Pattern</p>
          <p className={`text-2xl font-bold ${LABEL_COLORS[p.label] ?? "text-foreground"}`}>{labelDisplay}</p>
          <p className="text-sm text-muted-foreground mt-1">{LABEL_DESC[p.label] ?? ""}</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-semibold text-foreground mb-3">Class Probabilities</p>
        <div className="space-y-3">
          {Object.entries(p.all_scores)
            .sort(([, a], [, b]) => (b as number) - (a as number))
            .map(([label, score]) => {
              const pct = Math.round((score as number) * 100);
              const display = label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
              return (
                <div key={label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-foreground">{display}</span>
                    <span className={`text-sm font-bold ${LABEL_COLORS[label] ?? "text-foreground"}`}>{pct}%</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${LABEL_BG[label] ?? "bg-primary"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      <div className="bg-secondary/20 rounded-xl p-4 flex items-center gap-3">
        <CheckCircle className="w-4 h-4 text-muted-foreground shrink-0" />
        <p className="text-sm text-muted-foreground">
          Model confidence: <strong className="text-foreground">{Math.round(p.confidence * 100)}%</strong>
          &nbsp;· Random Forest classifier trained on 26 code metric features
        </p>
      </div>
    </div>
  );
}

// ─── FixesTab ─────────────────────────────────────────────────────────────────

const FIX_TEMPLATES: Record<string, { title: string; before: string; after: string; explanation: string }> = {
  sql_injection: {
    title: "SQL Injection Fix",
    before: 'query = "SELECT * FROM users WHERE id = " + user_id',
    after:  'query = "SELECT * FROM users WHERE id = %s"\ncursor.execute(query, (user_id,))',
    explanation: "Use parameterized queries or prepared statements. Never concatenate user input into SQL strings.",
  },
  command_injection: {
    title: "Command Injection Fix",
    before: 'os.system("ls " + user_input)',
    after:  'import subprocess\nsubprocess.run(["ls", user_input], check=True)',
    explanation: "Pass arguments as a list to subprocess.run() to prevent shell injection.",
  },
  hardcoded_secret: {
    title: "Hardcoded Secret Fix",
    before: 'api_key = "sk-prod-abc123xyz"',
    after:  'import os\napi_key = os.environ["API_KEY"]',
    explanation: "Store secrets in environment variables or a secrets manager, never in source code.",
  },
  xss: {
    title: "XSS Prevention",
    before: 'innerHTML = user_input',
    after:  'textContent = user_input  // or use sanitize-html library',
    explanation: "Use textContent instead of innerHTML, or sanitize user input before rendering.",
  },
  path_traversal: {
    title: "Path Traversal Fix",
    before: 'open(base_dir + user_filename)',
    after:  'safe = os.path.realpath(os.path.join(base_dir, user_filename))\nif not safe.startswith(base_dir): raise ValueError("Invalid path")\nopen(safe)',
    explanation: "Resolve and validate the absolute path before opening files from user input.",
  },
};

const DEBT_FIXES: Record<string, string> = {
  A: "Excellent — minimal action needed",
  B: "Good — address highest-priority refactoring suggestions",
  C: "Fair — break large functions, reduce nesting, fix dead code",
  D: "Poor — significant refactoring required; prioritize by issue count",
  F: "Critical — full code review needed; consider rewriting affected modules",
};

function FixesTab({ r }: { r: FullAnalysisResult }) {
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const fixes: Array<{ priority: "critical" | "high" | "medium" | "low"; category: string; title: string; before?: string; after?: string; explanation: string }> = [];

  // Security fixes
  (r.security?.findings ?? []).forEach((f: { vuln_type?: string; title?: string; severity?: string; snippet?: string; description?: string }) => {
    const tpl = FIX_TEMPLATES[f.vuln_type?.toLowerCase() ?? ""] ??
      FIX_TEMPLATES[Object.keys(FIX_TEMPLATES).find(k => (f.title ?? "").toLowerCase().includes(k.replace(/_/g, " "))) ?? ""];
    fixes.push({
      priority: (f.severity ?? "medium") as "critical" | "high" | "medium" | "low",
      category: "Security",
      title: tpl?.title ?? `Fix: ${f.title ?? f.vuln_type ?? "Security Issue"}`,
      before: tpl?.before ?? f.snippet,
      after: tpl?.after,
      explanation: tpl?.explanation ?? f.description ?? "Review the OWASP Top 10 for remediation guidance.",
    });
  });

  // Refactoring fixes
  (r.refactoring?.suggestions ?? []).forEach((s: { refactoring_type?: string; title?: string; priority?: string; description?: string; suggestion?: string }) => {
    fixes.push({
      priority: (s.priority ?? "medium") as "critical" | "high" | "medium" | "low",
      category: "Refactoring",
      title: s.title ?? s.refactoring_type ?? "Refactoring Needed",
      explanation: s.suggestion ?? s.description ?? "Apply the suggested refactoring pattern.",
    });
  });

  // Dead code
  if ((r.dead_code?.dead_line_count ?? 0) > 0) {
    fixes.push({
      priority: "low",
      category: "Dead Code",
      title: `Remove ${r.dead_code!.dead_line_count} dead code lines`,
      explanation: `${r.dead_code!.dead_line_count} unused lines detected. Remove them to reduce maintenance burden and improve clarity.`,
    });
  }

  // Debt
  const rating = r.technical_debt?.overall_rating;
  if (rating && rating !== "A" && rating !== "B") {
    fixes.push({
      priority: rating === "F" || rating === "D" ? "high" : "medium",
      category: "Tech Debt",
      title: `Address Technical Debt (Rating: ${rating})`,
      explanation: DEBT_FIXES[rating] ?? "Reduce technical debt by addressing the highest-impact issues first.",
    });
  }

  const PRI_CLS: Record<string, string> = {
    critical: "bg-destructive/20 text-destructive border border-destructive/40",
    high:     "bg-orange-500/20 text-orange-400 border border-orange-500/40",
    medium:   "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
    low:      "bg-primary/20 text-primary border border-primary/40",
  };

  if (fixes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <CheckCircle className="w-12 h-12 text-emerald-400 mb-3" />
        <p className="text-lg font-semibold text-foreground">No fixes needed</p>
        <p className="text-sm text-muted-foreground mt-1">All checks passed — this code looks great!</p>
      </div>
    );
  }

  const sorted = [...fixes].sort((a, b) => {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    return (order[a.priority] ?? 4) - (order[b.priority] ?? 4);
  });

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{sorted.length} fix recommendation{sorted.length !== 1 ? "s" : ""} — sorted by priority</p>
      {sorted.map((fix, i) => (
        <div key={i} className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="flex items-center gap-3 px-5 py-3 bg-secondary/20 border-b border-border">
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase shrink-0 ${PRI_CLS[fix.priority]}`}>{fix.priority}</span>
            <span className="text-xs text-muted-foreground font-medium">{fix.category}</span>
            <span className="font-semibold text-foreground text-sm flex-1">{fix.title}</span>
          </div>
          <div className="px-5 py-4 space-y-3">
            <p className="text-sm text-muted-foreground">{fix.explanation}</p>
            {(fix.before || fix.after) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {fix.before && (
                  <div>
                    <div className="text-[10px] font-semibold text-destructive/80 uppercase tracking-wide mb-1.5">Before (problematic)</div>
                    <pre className="bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2.5 text-xs font-mono text-foreground overflow-x-auto whitespace-pre-wrap">{fix.before}</pre>
                  </div>
                )}
                {fix.after && (
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wide">After (fixed)</div>
                      <button
                        className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => {
                          navigator.clipboard.writeText(fix.after!).catch(() => {});
                          setCopiedIdx(i);
                          setTimeout(() => setCopiedIdx(null), 2000);
                        }}
                      >
                        {copiedIdx === i ? <Check className="w-3 h-3 text-emerald-400" /> : <ClipboardCopy className="w-3 h-3" />}
                        {copiedIdx === i ? "Copied!" : "Copy"}
                      </button>
                    </div>
                    <pre className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2.5 text-xs font-mono text-foreground overflow-x-auto whitespace-pre-wrap">{fix.after}</pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: "overview",     label: "Overview",      Icon: Eye },
  { id: "security",    label: "Security",      Icon: Shield },
  { id: "complexity",  label: "Complexity",    Icon: Gauge },
  { id: "bugs",        label: "Bug Risk",      Icon: Bug },
  { id: "pattern",     label: "Pattern",       Icon: Brain },
  { id: "clones",      label: "Clones",        Icon: Copy },
  { id: "refactoring", label: "Refactoring",   Icon: Wrench },
  { id: "deadcode",    label: "Dead Code",     Icon: FileX },
  { id: "debt",        label: "Debt",          Icon: CreditCard },
  { id: "docs",        label: "Docs",          Icon: BookOpen },
  { id: "performance", label: "Performance",   Icon: Zap },
  { id: "deps",        label: "Dependencies",  Icon: Network },
  { id: "readability", label: "Readability",   Icon: AlignLeft },
  { id: "fixes",       label: "Fix Guide",     Icon: Lightbulb },
];

const ReviewDetail = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams<{ id: string }>();

  // Try: router state → history by ID → null (show mock)
  const result: FullAnalysisResult | null =
    location.state?.result ?? (id ? getEntry(id)?.result ?? null : null);

  // ── Feedback state ──
  const [feedbacks, setFeedbacks] = useState<Record<string, FindingFeedback>>(
    () => (result ? loadFeedback(result.filename) : {})
  );
  const [feedbackModal, setFeedbackModal] = useState<{ key: string; title: string } | null>(null);
  const [fbStatus, setFbStatus] = useState<FindingFeedback["status"]>("accepted");
  const [fbReason, setFbReason] = useState("");

  const openFeedbackModal = (key: string, title: string) => {
    const existing = feedbacks[key];
    setFbStatus(existing?.status ?? "accepted");
    setFbReason(existing?.reasoning ?? "");
    setFeedbackModal({ key, title });
  };

  const submitFeedback = () => {
    if (!feedbackModal || !result) return;
    const session = getSession();
    const fb: FindingFeedback = {
      status: fbStatus,
      reasoning: fbReason.trim(),
      reviewer: session?.name ?? "Reviewer",
      at: new Date().toISOString(),
    };
    const updated = { ...feedbacks, [feedbackModal.key]: fb };
    setFeedbacks(updated);
    persistFeedback(result.filename, updated);
    toast.success(`Issue marked as ${FB_LABEL[fbStatus]}`, {
      description: fbReason.trim() ? `"${fbReason.trim()}"` : undefined,
    });
    setFeedbackModal(null);
    setFbReason("");
  };

  const handleExport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url,
      download: `intellicode-report-${Date.now()}.json`,
    });
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Report exported as JSON");
  };

  const handleExportHTML = () => {
    if (!result) return;
    const sev = (s: string) => ({ critical: "#ef4444", high: "#f97316", medium: "#eab308", low: "#3b82f6", info: "#6b7280" }[s] ?? "#6b7280");
    const row = (label: string, value: string) =>
      `<tr><td style="padding:6px 12px;color:#94a3b8;font-size:13px">${label}</td><td style="padding:6px 12px;font-size:13px;font-weight:600">${value}</td></tr>`;
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>IntelliCode Report — ${result.filename}</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;margin:0;padding:32px}
  h1{color:#fff;margin-bottom:4px}h2{color:#a78bfa;font-size:15px;margin:24px 0 8px}
  .badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:11px;font-weight:700;text-transform:uppercase}
  .card{background:#1e2330;border-radius:12px;padding:16px;margin-bottom:12px;border-left:4px solid}
  table{width:100%;border-collapse:collapse}td{border-bottom:1px solid #2d3748}
  .score-ring{display:inline-flex;align-items:center;justify-content:center;width:64px;height:64px;border-radius:50%;border:4px solid #a78bfa;font-size:22px;font-weight:700;color:#a78bfa}
  .meta{color:#64748b;font-size:12px;margin-top:4px}
  .finding{background:#1e2330;border-radius:8px;padding:12px;margin-bottom:8px;border-left:4px solid}
  pre{background:#0d1117;border-radius:6px;padding:10px;overflow-x:auto;font-size:12px;color:#e2e8f0}
  @media print{body{background:#fff;color:#000}.card,.finding{background:#f8f9fa}}
</style>
</head>
<body>
<div style="display:flex;align-items:center;gap:24px;margin-bottom:24px">
  <div class="score-ring">${result.overall_score}</div>
  <div>
    <h1 style="margin:0">${result.filename}</h1>
    <div class="meta">${result.language} · ${result.duration_seconds}s · ${new Date().toLocaleString()}</div>
    <div style="margin-top:8px">
      <span class="badge" style="background:${{ critical:"#7f1d1d",action_required:"#78350f",clean:"#14532d" }[result.status] ?? "#1e2330"};color:${{ critical:"#fca5a5",action_required:"#fcd34d",clean:"#86efac" }[result.status] ?? "#e2e8f0"}">${result.status.replace("_"," ").toUpperCase()}</span>
    </div>
  </div>
</div>
<p style="color:#94a3b8;margin-bottom:24px">${result.summary}</p>

<h2>Overview</h2>
<table>
  ${row("Overall Score", `${result.overall_score}/100`)}
  ${row("Security Issues", result.security?.summary ? `${result.security.summary.total ?? 0} (${result.security.summary.critical ?? 0} critical, ${result.security.summary.high ?? 0} high)` : "N/A")}
  ${row("Complexity", result.complexity ? `${Math.round(result.complexity.score ?? 0)}/100 — Grade ${result.complexity.grade ?? "N/A"}` : "N/A")}
  ${row("Bug Risk", result.bug_prediction ? `${(result.bug_prediction.risk_level ?? "unknown").toUpperCase()} (${Math.round((result.bug_prediction.bug_probability ?? 0) * 100)}% probability)` : "N/A")}
  ${row("Code Clones", result.clones ? `${result.clones.clones?.length ?? 0} blocks (${((result.clones.clone_rate ?? 0) * 100).toFixed(0)}% rate)` : "N/A")}
  ${row("Dead Lines", result.dead_code ? `${result.dead_code.dead_line_count ?? 0} (${((result.dead_code.dead_ratio ?? 0) * 100).toFixed(1)}%)` : "N/A")}
  ${row("Technical Debt", result.technical_debt ? `${result.technical_debt.total_debt_minutes ?? 0} min — Rating ${result.technical_debt.overall_rating ?? "N/A"}` : "N/A")}
</table>

${(result.security?.findings?.length ?? 0) > 0 ? `
<h2>Security Findings</h2>
${result.security!.findings.map((f: { severity: string; title: string; description: string; lineno: number; cwe: string; snippet?: string }) => `
<div class="finding" style="border-color:${sev(f.severity)}">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    <span class="badge" style="background:${sev(f.severity)}20;color:${sev(f.severity)}">${f.severity.toUpperCase()}</span>
    <strong>${f.title}</strong>
    <span style="color:#64748b;font-size:12px;margin-left:auto">${f.cwe} · line ${f.lineno}</span>
  </div>
  <p style="color:#94a3b8;font-size:13px;margin:0 0 6px">${f.description}</p>
  ${f.snippet ? `<pre>${f.snippet.replace(/</g, "&lt;")}</pre>` : ""}
</div>`).join("")}` : ""}

${(result.bug_prediction?.risk_factors?.length ?? 0) > 0 ? `
<h2>Bug Risk Factors</h2>
<ul style="color:#94a3b8;font-size:13px;padding-left:20px">${result.bug_prediction!.risk_factors.map((f: string) => `<li>${f}</li>`).join("")}</ul>` : ""}

<div class="meta" style="margin-top:32px;text-align:center">Generated by IntelliCode Review · ${new Date().toLocaleString()}</div>
</body></html>`;
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url,
      download: `intellicode-report-${result.filename.replace(/[^a-z0-9]/gi, "_")}.html`,
    });
    a.click();
    URL.revokeObjectURL(url);
    toast.success("HTML report exported — open in browser to print as PDF");
  };

  const handleExportMarkdown = () => {
    if (!result) return;
    const lines: string[] = [];
    const pct = result.overall_score ?? 0;
    const bar = (v: number, max = 100) => {
      const filled = Math.round((v / max) * 20);
      return `[${"█".repeat(filled)}${"░".repeat(20 - filled)}] ${v}${max === 100 ? "/100" : ""}`;
    };
    lines.push(`# IntelliCode Analysis Report`);
    lines.push(`\n**File:** \`${result.filename}\`  `);
    lines.push(`**Language:** ${result.language}  `);
    lines.push(`**Analyzed:** ${new Date().toLocaleString()}  `);
    lines.push(`**Duration:** ${result.duration_seconds}s\n`);
    lines.push(`## Overall Quality Score\n\n${bar(pct)}\n`);
    lines.push(`## Metrics\n`);
    lines.push(`| Metric | Value |`);
    lines.push(`|---|---|`);
    lines.push(`| Complexity | ${result.complexity?.score ?? "N/A"} |`);
    lines.push(`| Bug Probability | ${result.bug_prediction ? Math.round((result.bug_prediction.bug_probability ?? 0) * 100) + "%" : "N/A"} |`);
    lines.push(`| Bug Risk Level | ${result.bug_prediction?.risk_level ?? "N/A"} |`);
    lines.push(`| Security Findings | ${result.security?.summary?.total ?? 0} |`);
    lines.push(`| Critical | ${result.security?.summary?.critical ?? 0} |`);
    lines.push(`| Tech Debt | ${result.technical_debt?.total_debt_minutes ?? 0} min |`);
    lines.push(`| Debt Rating | ${result.technical_debt?.overall_rating ?? "N/A"} |`);
    lines.push(`| Clone Rate | ${result.clones?.clone_rate != null ? (result.clones.clone_rate * 100).toFixed(1) : 0}% |`);
    lines.push(`| Dead Code Lines | ${result.dead_code?.dead_line_count ?? 0} |`);
    if (result.readability?.overall_score != null) lines.push(`| Readability | ${result.readability.overall_score}/100 |`);
    if (result.docs?.average_quality != null) lines.push(`| Doc Quality | ${result.docs.average_quality}/100 |`);
    lines.push(``);
    if ((result.security?.findings?.length ?? 0) > 0) {
      lines.push(`## Security Findings\n`);
      result.security!.findings.forEach((f: { severity?: string; title?: string; vuln_type?: string; description?: string; snippet?: string }) => {
        lines.push(`### [${(f.severity ?? "medium").toUpperCase()}] ${f.title ?? f.vuln_type ?? "Finding"}`);
        if (f.description) lines.push(`\n${f.description}\n`);
        if (f.snippet) lines.push(`\`\`\`python\n${f.snippet}\n\`\`\``);
      });
    }
    if ((result.bug_prediction?.risk_factors?.length ?? 0) > 0) {
      lines.push(`## Bug Risk Factors\n`);
      result.bug_prediction!.risk_factors.forEach((rf: string) => lines.push(`- ${rf}`));
      lines.push(``);
    }
    if ((result.refactoring?.suggestions?.length ?? 0) > 0) {
      lines.push(`## Refactoring Suggestions\n`);
      result.refactoring!.suggestions.forEach((s: { priority?: string; title?: string; description?: string }) => {
        lines.push(`- **[${s.priority ?? "medium"}]** ${s.title ?? "Suggestion"}${s.description ? ": " + s.description : ""}`);
      });
      lines.push(``);
    }
    lines.push(`---\n_Generated by IntelliCode Review_`);
    const md = lines.join("\n");
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url,
      download: `intellicode-report-${result.filename.replace(/[^a-z0-9]/gi, "_")}.md`,
    });
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Markdown report exported");
  };

  const handleExportPDF = () => {
    if (!result) return;
    // Build a minimal print-friendly HTML page and trigger browser print → Save as PDF
    const score = result.overall_score ?? 0;
    const status = score >= 90 ? "Clean" : score >= 75 ? "Good" : score >= 60 ? "Action Required" : "Critical";
    const secFindings = result.security?.findings ?? [];
    const findingsHtml = secFindings.length
      ? secFindings.map((f: { severity?: string; title?: string; description?: string }) =>
          `<tr><td style="color:${f.severity === "critical" ? "#ef4444" : f.severity === "high" ? "#f97316" : "#eab308"};font-weight:600;padding:6px 10px">${(f.severity ?? "").toUpperCase()}</td><td style="padding:6px 10px">${f.title ?? "Finding"}</td><td style="padding:6px 10px;color:#94a3b8;font-size:12px">${f.description ?? ""}</td></tr>`
        ).join("")
      : `<tr><td colspan="3" style="padding:6px 10px;color:#22c55e">No security issues found</td></tr>`;

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/><title>IntelliCode Report — ${result.filename}</title>
<style>body{font-family:system-ui,sans-serif;color:#e2e8f0;background:#0b1120;margin:0;padding:32px}
h1{font-size:22px;margin-bottom:4px}h2{font-size:15px;color:#94a3b8;margin:24px 0 8px}
.score{font-size:64px;font-weight:700;color:${score >= 75 ? "#22d3ee" : score >= 60 ? "#eab308" : "#ef4444"}}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border:1px solid #1e293b;padding:6px 10px;text-align:left}
th{background:#1e293b;color:#94a3b8;font-weight:500}@media print{body{-webkit-print-color-adjust:exact}}
</style></head><body>
<h1>IntelliCode Review — ${result.filename}</h1>
<div style="color:#94a3b8;font-size:13px;margin-bottom:24px">${new Date().toLocaleString()} · Status: <strong style="color:${score >= 75 ? "#22d3ee" : "#ef4444"}">${status}</strong></div>
<div class="score">${score}<span style="font-size:24px;color:#94a3b8">/100</span></div>
<h2>Metrics</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Complexity Score</td><td>${result.complexity?.score ?? "N/A"}</td></tr>
<tr><td>Bug Probability</td><td>${result.bug_prediction ? Math.round((result.bug_prediction.bug_probability ?? 0) * 100) + "%" : "N/A"}</td></tr>
<tr><td>Security Findings</td><td>${result.security?.summary?.total ?? 0}</td></tr>
<tr><td>Critical Issues</td><td>${result.security?.summary?.critical ?? 0}</td></tr>
<tr><td>Tech Debt</td><td>${result.technical_debt?.total_debt_minutes ?? 0} min (${result.technical_debt?.overall_rating ?? "N/A"})</td></tr>
<tr><td>Clone Rate</td><td>${result.clones?.clone_rate != null ? (result.clones.clone_rate * 100).toFixed(1) : 0}%</td></tr>
<tr><td>Dead Code Lines</td><td>${result.dead_code?.dead_line_count ?? 0}</td></tr>
${result.readability ? `<tr><td>Readability</td><td>${result.readability.overall_score}/100</td></tr>` : ""}
${result.docs ? `<tr><td>Doc Quality</td><td>${result.docs.average_quality}/100</td></tr>` : ""}
</tbody></table>
<h2>Security Findings</h2>
<table><thead><tr><th>Severity</th><th>Title</th><th>Description</th></tr></thead><tbody>${findingsHtml}</tbody></table>
<div style="margin-top:32px;color:#475569;font-size:11px">Generated by IntelliCode Review</div>
</body></html>`;

    const win = window.open("", "_blank", "width=900,height=700");
    if (!win) { toast.error("Popup blocked — allow popups to export PDF"); return; }
    win.document.write(html);
    win.document.close();
    win.onload = () => { win.print(); };
    toast.success("Print dialog opened — choose 'Save as PDF'");
  };

  // ── Fallback: no real result (e.g. /reviews/1 accessed directly) ──
  if (!result) {
    return (
      <div className="min-h-screen bg-background">
        <AppNavigation />
        <main className="container mx-auto px-4 py-8 max-w-5xl">
          <div className="bg-card border border-border rounded-xl p-8 mb-6 flex flex-col items-center gap-4 text-center">
            <div className="text-4xl">🔍</div>
            <h2 className="text-xl font-semibold text-foreground">Review not found</h2>
            <p className="text-sm text-muted-foreground max-w-sm">
              This review no longer exists or was never saved.{" "}
              <span className="text-primary cursor-pointer underline" onClick={() => navigate("/submit")}>
                Submit code
              </span>{" "}
              to generate a new analysis.
            </p>
          </div>
          <Button variant="outline" onClick={() => navigate("/reviews")}>← Back to Reviews</Button>
        </main>
      </div>
    );
  }

  // ── Real result view ──
  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-6 gap-4">
          <div className="flex items-center gap-4 min-w-0">
            <ScoreCircle score={result.overall_score} label="Quality" size={76} />
            <div className="min-w-0">
              <div className="text-sm text-muted-foreground mb-1">
                <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>
                  Dashboard
                </span>
                <span className="mx-2">›</span>
                <span className="text-foreground truncate">{result.filename}</span>
              </div>
              <h1 className="text-2xl font-bold text-foreground">Analysis Report</h1>
              <p className="text-sm text-muted-foreground mt-1">
                {result.language} · {result.duration_seconds}s
              </p>
            </div>
          </div>
          <div className="flex gap-2 shrink-0 flex-wrap">
            <Button variant="outline" size="sm" onClick={() => navigate("/submit", { state: { filename: result.filename } })}>
              Re-analyze
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="w-4 h-4 mr-1" />
              JSON
            </Button>
            <Button variant="outline" size="sm" onClick={handleExportHTML}>
              <Download className="w-4 h-4 mr-1" />
              HTML
            </Button>
            <Button variant="outline" size="sm" onClick={handleExportMarkdown}>
              <Download className="w-4 h-4 mr-1" />
              Markdown
            </Button>
            <Button variant="outline" size="sm" onClick={handleExportPDF}>
              <Download className="w-4 h-4 mr-1" />
              PDF
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview">
          <div className="overflow-x-auto mb-6 -mx-1 px-1">
            <TabsList className="flex w-max gap-1 bg-secondary/40 p-1 rounded-xl">
              {TABS.map(({ id, label, Icon }) => (
                <TabsTrigger
                  key={id} value={id}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 shrink-0 rounded-lg data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          <div className="bg-card border border-border rounded-xl p-6 min-h-[300px]">
            <TabsContent value="overview">    <OverviewTab r={result} /></TabsContent>
            <TabsContent value="security">   <SecurityTab r={result} feedbacks={feedbacks} onReview={openFeedbackModal} /></TabsContent>
            <TabsContent value="complexity"> <ComplexityTab r={result} /></TabsContent>
            <TabsContent value="bugs">       <BugTab r={result} /></TabsContent>
            <TabsContent value="pattern">    <PatternTab r={result} /></TabsContent>
            <TabsContent value="clones">     <ClonesTab r={result} /></TabsContent>
            <TabsContent value="refactoring"><RefactoringTab r={result} /></TabsContent>
            <TabsContent value="deadcode">   <DeadCodeTab r={result} /></TabsContent>
            <TabsContent value="debt">       <DebtTab r={result} /></TabsContent>
            <TabsContent value="docs">       <DocsTab r={result} /></TabsContent>
            <TabsContent value="performance"><PerformanceTab r={result} /></TabsContent>
            <TabsContent value="deps">       <DepsTab r={result} /></TabsContent>
            <TabsContent value="readability"><ReadabilityTab r={result} /></TabsContent>
            <TabsContent value="fixes">       <FixesTab r={result} /></TabsContent>
          </div>
        </Tabs>

        <div className="flex items-center justify-between mt-6">
          <Button variant="outline" onClick={() => navigate("/dashboard")}>← Back to Dashboard</Button>
          <Button className="bg-gradient-primary" onClick={() => navigate("/submit")}>
            Analyze Another File
          </Button>
        </div>

        <AnalysisFeedbackBar result={result} />
      </main>

      {/* Reviewer Feedback Modal */}
      {feedbackModal && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setFeedbackModal(null)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-md w-full space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-foreground">Reviewer Decision</h3>
              <button onClick={() => setFeedbackModal(null)} className="text-muted-foreground hover:text-foreground">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-muted-foreground">
              Issue: <span className="text-foreground font-medium">{feedbackModal.title}</span>
            </p>

            {/* Decision buttons */}
            <div className="grid grid-cols-3 gap-2">
              {(["accepted", "rejected", "false_positive"] as const).map((s) => {
                const icons = { accepted: ThumbsUp, rejected: ThumbsDown, false_positive: Flag };
                const labels = { accepted: "Accept", rejected: "Reject", false_positive: "False Positive" };
                const active = {
                  accepted: "bg-green-500 text-white",
                  rejected: "bg-orange-500 text-white",
                  false_positive: "bg-muted text-foreground border-primary",
                };
                const inactive = "bg-secondary text-muted-foreground hover:bg-secondary/80";
                const Icon = icons[s];
                return (
                  <button
                    key={s}
                    onClick={() => setFbStatus(s)}
                    className={`flex flex-col items-center gap-1.5 py-3 px-2 rounded-lg border text-xs font-medium transition-all ${fbStatus === s ? active[s] : inactive}`}
                  >
                    <Icon className="w-4 h-4" />
                    {labels[s]}
                  </button>
                );
              })}
            </div>

            {/* Reasoning */}
            <div>
              <label className="text-sm text-foreground mb-1.5 block font-medium">
                Reasoning <span className="text-muted-foreground font-normal">(optional)</span>
              </label>
              <textarea
                value={fbReason}
                onChange={(e) => setFbReason(e.target.value)}
                rows={3}
                placeholder="Explain your decision..."
                className="w-full rounded-lg bg-input border border-border text-sm text-foreground p-3 resize-none focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>

            <div className="flex gap-3 justify-end">
              <Button variant="outline" size="sm" onClick={() => setFeedbackModal(null)}>Cancel</Button>
              <Button size="sm" className="bg-gradient-primary" onClick={submitFeedback}>
                Save Decision
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewDetail;
