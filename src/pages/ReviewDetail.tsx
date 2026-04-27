import { useState, useMemo, useEffect } from "react";
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
  Loader2,
  MapPin,
  Github,
  ExternalLink,
  ChevronDown,
  Info,
} from "lucide-react";
import { getEntry, getEntries } from "@/services/reviewHistory";
import { STORAGE_KEYS } from "@/constants/storage";
import { getSession } from "@/services/auth";
import { toast } from "sonner";
import type { FullAnalysisResult, OODInfo, ConformalInterval } from "@/types/analysis";
import { submitAnalysisFeedback, explainAnalysis, BASE_URL, type ExplainResult } from "@/services/api";
import { getGitHubToken, isGitHubConnected, listUserRepos, getRawFile, type GitHubRepo } from "@/services/github";

// ─── Feedback types ────────────────────────────────────────────────────────────

interface FindingFeedback {
  status: "accepted" | "rejected" | "false_positive";
  reasoning: string;
  reviewer: string;
  at: string;
}

const FEEDBACK_KEY = STORAGE_KEYS.feedback;

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
      toast.success(rating === "positive" ? "Thanks! Glad it was helpful." : "Thanks for the feedback — we'll use it to improve.");
    } catch {
      toast.warning("Feedback saved locally — could not reach backend.");
    } finally {
      setSent(true);
      setSubmitting(false);
    }
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

// ─── File score history sparkline ─────────────────────────────────────────────

function FileScoreHistory({ filename }: { filename: string }) {
  const history = getEntries()
    .filter((e) => e.filename === filename)
    .sort((a, b) => new Date(a.submittedAt).getTime() - new Date(b.submittedAt).getTime())
    .slice(-10);

  if (history.length < 2) return null;

  const scores = history.map((e) => e.overallScore);
  const minS = Math.min(...scores);
  const maxS = Math.max(...scores);
  const range = maxS - minS || 1;
  const W = 220, H = 48, PAD = 4;
  const pts = scores.map((s, i) => {
    const x = PAD + (i / (scores.length - 1)) * (W - PAD * 2);
    const y = PAD + ((maxS - s) / range) * (H - PAD * 2);
    return `${x},${y}`;
  }).join(" ");

  const latest = scores[scores.length - 1];
  const first = scores[0];
  const delta = latest - first;
  const deltaColor = delta >= 0 ? "text-emerald-400" : "text-red-400";
  const lineColor = delta >= 0 ? "#10b981" : "#ef4444";

  return (
    <div className="bg-secondary/30 border border-border rounded-xl px-4 py-3 flex items-center gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-muted-foreground mb-1">Score history — {history.length} analyses</p>
        <svg width={W} height={H} className="overflow-visible">
          <polyline fill="none" stroke={lineColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" points={pts} />
          {scores.map((s, i) => {
            const x = PAD + (i / (scores.length - 1)) * (W - PAD * 2);
            const y = PAD + ((maxS - s) / range) * (H - PAD * 2);
            return <circle key={i} cx={x} cy={y} r="3" fill={lineColor} />;
          })}
        </svg>
      </div>
      <div className="text-right shrink-0">
        <div className="text-2xl font-bold text-foreground">{latest}</div>
        <div className={`text-xs font-semibold ${deltaColor}`}>
          {delta >= 0 ? "+" : ""}{delta} pts
        </div>
        <div className="text-[10px] text-muted-foreground">vs first run</div>
      </div>
    </div>
  );
}

// ─── Quality Radar ────────────────────────────────────────────────────────────

function QualityRadar({ r }: { r: FullAnalysisResult }) {
  const secScore = Math.max(0, 100 - (r.security?.findings?.length ?? 0) * 8);
  const dims = [
    { label: "Security",   score: secScore },
    { label: "Complexity", score: r.complexity?.score ?? 0 },
    { label: "Bug Risk",   score: Math.round((1 - (r.bug_prediction?.bug_probability ?? 0)) * 100) },
    { label: "Readability",score: r.readability?.overall_score ?? 0 },
    { label: "Docs",       score: r.docs?.average_quality ?? 0 },
  ];
  const N = dims.length;
  const CX = 80, CY = 80, R = 60;
  const angle = (i: number) => (i / N) * 2 * Math.PI - Math.PI / 2;
  const pt = (i: number, radius: number) => ({
    x: CX + radius * Math.cos(angle(i)),
    y: CY + radius * Math.sin(angle(i)),
  });
  const gridLevels = [0.25, 0.5, 0.75, 1];
  const dataPoints = dims.map((d, i) => pt(i, (d.score / 100) * R));
  const dataPath = dataPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ") + "Z";

  return (
    <div className="bg-secondary/20 border border-border rounded-xl p-4 flex flex-col sm:flex-row items-center gap-5">
      <svg width={160} height={160} viewBox="0 0 160 160" className="shrink-0">
        {gridLevels.map((lvl) => {
          const pts = Array.from({ length: N }, (_, i) => pt(i, lvl * R));
          return (
            <path key={lvl}
              d={pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ") + "Z"}
              fill="none" stroke="hsl(var(--border))" strokeWidth="1" />
          );
        })}
        {dims.map((_, i) => {
          const end = pt(i, R);
          return <line key={i} x1={CX} y1={CY} x2={end.x} y2={end.y} stroke="hsl(var(--border))" strokeWidth="1" />;
        })}
        <path d={dataPath} fill="hsl(var(--primary))" fillOpacity="0.18" stroke="hsl(var(--primary))" strokeWidth="2" strokeLinejoin="round" />
        {dataPoints.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r="3" fill="hsl(var(--primary))" />
        ))}
        {dims.map((d, i) => {
          const labelR = R + 18;
          const p = { x: CX + labelR * Math.cos(angle(i)), y: CY + labelR * Math.sin(angle(i)) };
          return (
            <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="central"
              fontSize="8.5" fill="hsl(var(--muted-foreground))">{d.label}
            </text>
          );
        })}
      </svg>
      <div className="space-y-2 flex-1 w-full">
        {dims.map((d) => {
          const color = d.score >= 75 ? "bg-emerald-500" : d.score >= 50 ? "bg-yellow-500" : "bg-red-500";
          return (
            <div key={d.label} className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-20 shrink-0">{d.label}</span>
              <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${color}`} style={{ width: `${d.score}%` }} />
              </div>
              <span className="text-xs font-semibold text-foreground w-6 text-right shrink-0">{Math.round(d.score)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Tab bodies ───────────────────────────────────────────────────────────────

function OverviewTab({ r }: { r: FullAnalysisResult }) {
  // Delta vs immediately previous scan of the same file
  const prev = useMemo(() => {
    const entries = getEntries()
      .filter((e) => e.filename === r.filename)
      .sort((a, b) => new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime());
    return entries.length >= 2 ? entries[1] : null;
  }, [r.filename]);

  const prevResult = prev?.result ?? null;

  function Delta({ current, previous }: { current: number; previous: number | null }) {
    if (previous === null) return null;
    const d = current - previous;
    if (d === 0) return null;
    return (
      <span className={`text-[10px] font-bold ml-1 ${d > 0 ? "text-emerald-400" : "text-red-400"}`}>
        {d > 0 ? "+" : ""}{d}
      </span>
    );
  }

  return (
    <div className="space-y-6">
      <FileScoreHistory filename={r.filename} />
      <QualityRadar r={r} />
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
            delta: prevResult ? <Delta current={r.security?.summary?.total ?? 0} previous={prevResult.security?.summary?.total ?? 0} /> : null,
          },
          {
            label: "Bug Risk",
            value: (r.bug_prediction?.risk_level ?? "unknown").toUpperCase(),
            sub: `${Math.round((r.bug_prediction?.bug_probability ?? 0) * 100)}% probability`,
            cls: ({ low: "text-green-500", medium: "text-yellow-500", high: "text-orange-500", critical: "text-destructive" } as Record<string, string>)[r.bug_prediction?.risk_level ?? ""] ?? "text-foreground",
            border: ({ low: "border-l-green-500", medium: "border-l-yellow-500", high: "border-l-orange-500", critical: "border-l-destructive" } as Record<string, string>)[r.bug_prediction?.risk_level ?? ""] ?? "border-l-primary",
            delta: prevResult ? <Delta current={Math.round((r.bug_prediction?.bug_probability ?? 0) * 100)} previous={Math.round((prevResult.bug_prediction?.bug_probability ?? 0) * 100)} /> : null,
          },
          {
            label: "Clones Found",
            value: r.clones?.clones?.length ?? 0,
            sub: `${((r.clones?.clone_rate ?? 0) * 100).toFixed(0)}% duplication`,
            cls: "text-foreground",
            border: (r.clones?.clones?.length ?? 0) > 0 ? "border-l-yellow-500" : "border-l-green-500",
            delta: prevResult ? <Delta current={r.clones?.clones?.length ?? 0} previous={prevResult.clones?.clones?.length ?? 0} /> : null,
          },
          {
            label: "Dead Code",
            value: `${r.dead_code?.dead_line_count ?? 0} lines`,
            sub: `${((r.dead_code?.dead_ratio ?? 0) * 100).toFixed(1)}% of file`,
            cls: "text-foreground",
            border: (r.dead_code?.dead_line_count ?? 0) > 0 ? "border-l-primary" : "border-l-green-500",
            delta: prevResult ? <Delta current={r.dead_code?.dead_line_count ?? 0} previous={prevResult.dead_code?.dead_line_count ?? 0} /> : null,
          },
        ].map((card) => (
          <div key={card.label} className={`bg-secondary/40 rounded-xl p-4 text-center border-l-4 ${card.border}`}>
            <p className={`text-2xl font-bold ${card.cls}`}>{card.value}{card.delta}</p>
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

// ─── Decision Badge ───────────────────────────────────────────────────────────

const DECISION_STYLES: Record<string, string> = {
  fix_now:         "bg-red-500/15 text-red-400 border border-red-500/30",
  review_manually: "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  low_priority:    "bg-blue-500/15 text-blue-400 border border-blue-500/30",
  unreliable:      "bg-secondary text-muted-foreground border border-border",
};

function DecisionBadge({ decision }: { decision?: { action: string; label: string; explanation: string } }) {
  if (!decision) return null;
  return (
    <span
      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap ${DECISION_STYLES[decision.action] ?? DECISION_STYLES.unreliable}`}
      title={decision.explanation}
    >
      {decision.label}
    </span>
  );
}

// ─── OOD / Conformal Uncertainty Banner ───────────────────────────────────────

function OODBanner({ ood, reason, conformal }: {
  ood?: OODInfo;
  reason?: string;
  conformal?: ConformalInterval;
}) {
  const isLow = ood?.low_confidence ?? false;
  if (!isLow && !conformal) return null;
  return (
    <div className="rounded-xl border border-blue-500/30 bg-blue-500/8 p-3 space-y-2 mb-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Brain className="w-4 h-4 text-blue-400 shrink-0" />
        <span className="text-xs font-semibold text-blue-400">Uncertainty estimate</span>
        {isLow && (
          <span className="text-[10px] bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-2 py-0.5 rounded-full font-medium">
            Low confidence — OOD input ({ood!.sigma_distance.toFixed(1)}&sigma;)
          </span>
        )}
      </div>
      {reason && (
        <p className="text-xs text-muted-foreground italic">{reason}</p>
      )}
      {conformal && (
        <div className="text-xs text-muted-foreground space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span>
              {Math.round(conformal.coverage_level * 100)}% prediction interval:
            </span>
            <span className="font-mono text-foreground">
              [{(conformal.lower * 100).toFixed(0)}%, {(conformal.upper * 100).toFixed(0)}%]
            </span>
            {conformal.is_fallback && (
              <span className="text-[10px] text-muted-foreground/60 italic">
                (default quantile — run calibration for tighter bounds)
              </span>
            )}
            {!conformal.is_fallback && (
              <span className="text-[10px] text-muted-foreground/60 italic">
                calibrated on {conformal.n_cal} held-out samples
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Trust Banner ─────────────────────────────────────────────────────────────

function TrustBanner({ r }: { r: FullAnalysisResult }) {
  const ts = r.trust_summary;
  if (!ts) return null;
  if (ts.overall_reliable && ts.language_in_distribution) return null;

  return (
    <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/8 p-4 space-y-2 mb-4">
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0" />
        <span className="text-sm font-semibold text-yellow-400">Model reliability notice</span>
        {r.model_version && (
          <span className="text-xs text-muted-foreground ml-auto">v{r.model_version}</span>
        )}
      </div>
      <div className="space-y-1">
        {ts.items.map((item) => (
          <div key={item.model} className="flex items-start gap-2 text-xs">
            <span className={`shrink-0 font-semibold w-20 ${
              item.reliability === "high" ? "text-green-400"
              : item.reliability === "medium" ? "text-yellow-400"
              : "text-red-400"
            }`}>
              {item.model}:
            </span>
            <span className="text-muted-foreground">{item.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ExploitabilityBar({ score }: { score: number }) {
  const color =
    score >= 8 ? "bg-destructive" :
    score >= 6 ? "bg-orange-500" :
    score >= 4 ? "bg-yellow-500" : "bg-green-500";
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-muted-foreground uppercase tracking-wide w-20 shrink-0">Exploitability</span>
      <div className="flex-1 bg-secondary rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full transition-all`} style={{ width: `${score * 10}%` }} />
      </div>
      <span className="text-[10px] font-bold text-foreground w-6 text-right">{score}/10</span>
    </div>
  );
}

// ─── Root cause descriptions (mirrors backend context_analyzer._ROOT_CAUSE_DESCRIPTIONS) ─
const ROOT_CAUSE_DESC: Record<string, string> = {
  sql_injection:             "String concatenation into SQL lets an attacker alter query logic and read or modify any data in the database.",
  command_injection:         "Shell metacharacters in user input allow arbitrary OS command execution on the server.",
  eval_injection:            "eval() / exec() on user-supplied input allows arbitrary Python code execution (full RCE).",
  code_injection:            "Dynamic code execution on user input allows arbitrary Python code execution (full RCE).",
  path_traversal:            "Unsanitized file paths let attackers escape the intended directory and read or overwrite arbitrary files.",
  insecure_deserialization:  "pickle.loads() on untrusted bytes executes arbitrary code during deserialization.",
  hardcoded_secret:          "API keys and passwords in source code are exposed to anyone with repo access or in compiled binaries.",
  hardcoded_credential:      "Credentials in source code are exposed to anyone with repo access or in compiled binaries.",
  ssrf:                      "User-controlled URLs in server-side requests allow attackers to reach internal services and metadata endpoints.",
  xss:                       "Unsanitized output reflected into HTML lets attackers inject scripts that run in other users' browsers.",
  weak_crypto:               "MD5/SHA-1 are broken for password hashing — offline brute-force cracks them in seconds.",
  weak_cryptography:         "Weak cryptographic algorithms provide insufficient protection against modern attacks.",
  debug_enabled:             "Debug mode exposes stack traces, interactive consoles, and internal state to the network.",
  insecure_random:           "random.* is predictable — use secrets.* for tokens, session IDs, and cryptographic values.",
  open_redirect:             "Unvalidated redirect targets let attackers redirect users to malicious sites after login.",
  missing_auth:              "Endpoints without authentication checks are accessible to unauthenticated users.",
};

// Human-readable label for a vuln_type key
function vulnLabel(vuln_type: string): string {
  return vuln_type
    .split("_")
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// A single finding card (used inside groups and standalone)
function FindingCard({
  f,
  fb,
  onReview,
}: {
  f: FullAnalysisResult["security"]["findings"][0];
  fb?: FindingFeedback;
  onReview: (key: string, title: string) => void;
}) {
  const key = `${f.title}-${f.lineno}`;
  return (
    <div className={`bg-secondary/30 rounded-xl p-4 space-y-2.5 border-l-4 ${SEV_BORDER[f.severity] ?? "border-l-border"}`}>
      <div className="flex items-center gap-2 flex-wrap">
        <SevBadge sev={f.severity} />
        <DecisionBadge decision={f.decision} />
        {f.user_controlled && (
          <span className="text-[10px] font-bold bg-destructive/15 text-destructive px-2 py-0.5 rounded-full uppercase tracking-wide">
            user-controlled
          </span>
        )}
        <span className="font-semibold text-foreground text-sm">{f.title}</span>
        <span className="text-xs text-muted-foreground ml-auto">{f.cwe} · line {f.lineno}</span>
      </div>

      {f.context_sentence ? (
        <p className="text-sm text-foreground/90 leading-relaxed">{f.context_sentence}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{f.description}</p>
      )}

      {f.taint_path && f.taint_source && !["unknown", "constant", "internal"].includes(f.taint_source) && (
        <div className="bg-secondary/60 rounded px-3 py-2">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Data flow</p>
          <p className="text-xs font-mono text-foreground/80 break-all">{f.taint_path}</p>
        </div>
      )}

      {f.snippet && (
        <pre className="bg-destructive/10 border border-destructive/20 rounded p-3 text-xs font-mono text-foreground overflow-x-auto">
          {f.snippet}
        </pre>
      )}

      {f.exploitability !== undefined && <ExploitabilityBar score={f.exploitability} />}

      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs text-muted-foreground">
          Confidence: {Math.round(f.confidence * 100)}%
          {f.decision && f.decision.action !== "unreliable" && (
            <span className="ml-2 italic">{f.decision.explanation}</span>
          )}
        </p>
        {fb ? (
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${FB_BADGE[fb.status]}`}>
              {FB_LABEL[fb.status]}
            </span>
            <span className="text-xs text-muted-foreground">by {fb.reviewer}</span>
            <button className="text-xs text-primary hover:underline" onClick={() => onReview(key, f.title)}>Edit</button>
          </div>
        ) : (
          <button className="flex items-center gap-1 text-xs text-primary hover:underline" onClick={() => onReview(key, f.title)}>
            <MessageSquare className="w-3 h-3" />
            Add Review
          </button>
        )}
      </div>
      {fb?.reasoning && (
        <p className="text-xs text-muted-foreground italic border-l-2 border-border pl-2">"{fb.reasoning}"</p>
      )}
    </div>
  );
}

// A grouped finding block: header with root cause + collapsible instances
function FindingGroup({
  vulnType,
  findings,
  feedbacks,
  onReview,
}: {
  vulnType: string;
  findings: FullAnalysisResult["security"]["findings"];
  feedbacks: Record<string, FindingFeedback>;
  onReview: (key: string, title: string) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const maxSev = findings.reduce<string>((best, f) => {
    const order = ["critical", "high", "medium", "low"];
    return order.indexOf(f.severity) < order.indexOf(best) ? f.severity : best;
  }, "low");
  const userCtrl = findings.filter(f => f.user_controlled).length;
  const rootCause = ROOT_CAUSE_DESC[vulnType];

  // If only one finding, skip the group wrapper
  if (findings.length === 1) {
    const key = `${findings[0].title}-${findings[0].lineno}`;
    return <FindingCard f={findings[0]} fb={feedbacks[key]} onReview={onReview} />;
  }

  return (
    <div className={`rounded-xl border border-border overflow-hidden border-l-4 ${SEV_BORDER[maxSev] ?? "border-l-border"}`}>
      {/* Group header */}
      <button
        className="w-full flex items-center gap-3 px-4 py-3 bg-secondary/40 hover:bg-secondary/60 transition-colors text-left"
        onClick={() => setExpanded(v => !v)}
      >
        <SevBadge sev={maxSev} />
        <span className="font-semibold text-foreground text-sm">{vulnLabel(vulnType)}</span>
        <span className="text-xs bg-secondary border border-border rounded-full px-2 py-0.5 text-muted-foreground font-medium">
          {findings.length} instance{findings.length > 1 ? "s" : ""}
        </span>
        {userCtrl > 0 && (
          <span className="text-[10px] font-bold bg-destructive/15 text-destructive px-2 py-0.5 rounded-full uppercase tracking-wide">
            {userCtrl} user-controlled
          </span>
        )}
        <ChevronDown className={`w-4 h-4 text-muted-foreground ml-auto shrink-0 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>

      {/* Root cause explanation — always visible */}
      {rootCause && (
        <div className="px-4 py-2 bg-secondary/20 border-b border-border">
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Root cause: </span>{rootCause}
          </p>
        </div>
      )}

      {/* Collapsible instances */}
      {expanded && (
        <div className="p-3 space-y-3">
          {findings.map((f, i) => {
            const key = `${f.title}-${f.lineno}`;
            return <FindingCard key={key} f={f} fb={feedbacks[key]} onReview={onReview} />;
          })}
        </div>
      )}

      {!expanded && (
        <div className="px-4 py-2 text-xs text-muted-foreground">
          Lines: {findings.map(f => f.lineno).join(", ")} — click to expand
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
  const fps = r.security.false_positives ?? [];
  const enriched = r.security.taint_enriched ?? false;
  const [showFPs, setShowFPs] = useState(false);

  if (findings.length === 0 && fps.length === 0)
    return <Empty msg="No security vulnerabilities detected." />;

  const sevOrder = ["critical", "high", "medium", "low"];
  const priorityOf = (f: typeof findings[0]) =>
    f.decision?.priority ?? (sevOrder.indexOf(f.severity) + 1);

  // Group by vuln_type, sort within each group by exploitability desc
  const groups = useMemo(() => {
    const map = new Map<string, typeof findings>();
    for (const f of findings) {
      const key = f.vuln_type ?? "unknown";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(f);
    }
    // Sort instances within each group
    for (const [, grp] of map) {
      grp.sort((a, b) => {
        const expDiff = (b.exploitability ?? 5) - (a.exploitability ?? 5);
        return expDiff !== 0 ? expDiff : priorityOf(a) - priorityOf(b);
      });
    }
    // Sort groups by their worst finding
    return [...map.entries()].sort(([, a], [, b]) => {
      const topA = a[0];
      const topB = b[0];
      const expDiff = (topB.exploitability ?? 5) - (topA.exploitability ?? 5);
      return expDiff !== 0 ? expDiff : priorityOf(topA) - priorityOf(topB);
    });
  }, [findings]);

  const userControlledCount = findings.filter(f => f.user_controlled).length;
  const uniqueVulnTypes = groups.length;

  return (
    <div className="space-y-4">
      <TrustBanner r={r} />
      <OODBanner
        ood={r.security.ood}
        reason={r.security.low_confidence_reason}
        conformal={r.security.conformal_interval}
      />

      {/* Re-analyze hint for old entries without taint data */}
      {!enriched && findings.length > 0 && (
        <div className="rounded-lg border border-border bg-secondary/20 px-4 py-3 flex items-center gap-3">
          <span className="text-xs text-muted-foreground">
            Taint-flow analysis not available for this entry.
          </span>
          <a href="/" className="text-xs text-primary hover:underline ml-auto shrink-0">
            Re-analyze to enable
          </a>
        </div>
      )}

      {/* Taint analysis summary banner */}
      {enriched && (
        <div className="rounded-lg border border-primary/20 bg-primary/5 px-4 py-3 flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-primary uppercase tracking-widest">Taint Analysis</span>
          {userControlledCount > 0 && (
            <span className="text-xs bg-destructive/15 text-destructive px-2 py-0.5 rounded-full font-medium">
              {userControlledCount} confirmed user-controlled
            </span>
          )}
          {fps.length > 0 && (
            <span className="text-xs bg-secondary text-muted-foreground px-2 py-0.5 rounded-full">
              {fps.length} likely false positive{fps.length > 1 ? "s" : ""} filtered
            </span>
          )}
          <span className="text-xs text-muted-foreground ml-auto">
            {uniqueVulnTypes} pattern type{uniqueVulnTypes > 1 ? "s" : ""} · {findings.length} instance{findings.length > 1 ? "s" : ""} · sorted by exploitability
          </span>
        </div>
      )}

      <div className="flex gap-3 text-sm flex-wrap mb-2">
        {(["critical", "high", "medium", "low"] as const).map((sev) => (
          <span key={sev} className="flex items-center gap-1">
            <SevBadge sev={sev} />
            <span className="text-muted-foreground">{summary[sev]}</span>
          </span>
        ))}
      </div>

      {/* Grouped findings */}
      {groups.map(([vulnType, grpFindings]) => (
        <FindingGroup
          key={vulnType}
          vulnType={vulnType}
          findings={grpFindings}
          feedbacks={feedbacks}
          onReview={onReview}
        />
      ))}

      {/* False positives section */}
      {fps.length > 0 && (
        <div className="mt-2">
          <button
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 mb-2"
            onClick={() => setShowFPs(v => !v)}
          >
            <ChevronDown className={`w-3 h-3 transition-transform ${showFPs ? "rotate-180" : ""}`} />
            {showFPs ? "Hide" : "Show"} {fps.length} filtered false positive{fps.length > 1 ? "s" : ""}
          </button>
          {showFPs && fps.map((f, i) => (
            <div key={`${f.title}-${f.lineno}`} className="bg-secondary/15 rounded-lg p-3 mb-2 border-l-4 border-l-border opacity-60">
              <div className="flex items-center gap-2">
                <SevBadge sev={f.severity} />
                <span className="text-xs text-muted-foreground line-through">{f.title}</span>
                <span className="text-[10px] bg-secondary px-2 py-0.5 rounded-full text-muted-foreground ml-auto">false positive</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">{f.context_sentence || "Argument is a constant or internal value — not user-controlled."}</p>
            </div>
          ))}
        </div>
      )}
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
              <div key={fi.name ?? i} className="bg-secondary/20 rounded-lg p-3 flex items-center gap-3 text-sm">
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
  const displayProb = b.probability_adjusted ?? b.bug_probability;
  const pct = Math.round(displayProb * 100);
  return (
    <div className="space-y-6">
      <OODBanner
        ood={b.ood}
        reason={b.low_confidence_reason}
        conformal={b.conformal_interval}
      />
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
          {b.ood?.low_confidence && b.probability_adjusted !== undefined && (
            <p className="text-xs text-muted-foreground italic">
              Adjusted probability: {Math.round(b.probability_adjusted * 100)}%
              (raw: {Math.round(b.bug_probability * 100)}%)
            </p>
          )}
        </div>
      </div>
      {b.risk_factors.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">Risk Factors</h3>
          <ul className="space-y-2">
            {b.risk_factors.map((f, i) => (
              <li key={`rf-${i}`} className="flex items-start gap-2 text-sm text-muted-foreground">
                <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
      {b.top_feature_importances && b.top_feature_importances.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-2">Top Model Features (XGBoost global importance)</h3>
          <div className="space-y-2">
            {b.top_feature_importances.map((fi: { feature: string; importance: number }, i: number) => {
              const maxImp = b.top_feature_importances[0]?.importance ?? 1;
              const pctWidth = Math.round((fi.importance / maxImp) * 100);
              return (
                <div key={fi.feature} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground font-mono">{fi.feature}</span>
                    <span className="text-muted-foreground">{(fi.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-orange-400 rounded-full" style={{ width: `${pctWidth}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-muted-foreground mt-2 italic">
            Global feature importance from trained model — not per-prediction attribution.
          </p>
        </div>
      )}
      {b.reliability_context && (
        <div className="rounded-md border border-border bg-muted/30 p-3 space-y-1">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">Model Reliability</h3>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="font-mono text-foreground">{(((b.reliability_context?.in_dist_auc ?? 0) * 100)).toFixed(0)}%</div>
              <div className="text-muted-foreground">In-dist AUC</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-foreground">{(((b.reliability_context?.temporal_cv_auc ?? 0) * 100)).toFixed(0)}%</div>
              <div className="text-muted-foreground">Temporal CV</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-foreground">{(((b.reliability_context?.lopo_auc ?? 0) * 100)).toFixed(0)}%</div>
              <div className="text-muted-foreground">Cross-project</div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-1">{b.reliability_context?.recommended_use}</p>
        </div>
      )}
      <DREDeltaPanel r={r} />
      <HighRiskFunctions r={r} />
    </div>
  );
}

function DREDeltaPanel({ r }: { r: FullAnalysisResult }) {
  const dre = r.dre;
  if (!dre) return null;

  const pct = Math.round(dre.risk_score * 100);
  const deltaPct = Math.round(dre.delta_contribution * 100);
  const changed = dre.top_delta_features?.filter((f) => Math.abs(f.delta) > 0.01) ?? [];

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Change Risk (vs previous submission)</h3>
        <span className="text-xs text-muted-foreground">{dre.model_type}</span>
      </div>
      <div className="flex items-center gap-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-foreground">{pct}%</div>
          <div className="text-xs text-muted-foreground">Delta risk score</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-foreground">+{deltaPct}pp</div>
          <div className="text-xs text-muted-foreground">Above static-only</div>
        </div>
      </div>
      {changed.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground font-medium">Most changed features:</p>
          {changed.slice(0, 3).map((f) => (
            <div key={f.feature} className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-mono">{f.feature}</span>
              <span className={f.delta > 0 ? "text-red-400" : "text-green-400"}>
                {f.delta > 0 ? "+" : ""}{f.delta.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const riskColors: Record<string, string> = {
  critical: "text-red-500 bg-red-500/10 border-red-500/30",
  high:     "text-orange-500 bg-orange-500/10 border-orange-500/30",
  medium:   "text-yellow-500 bg-yellow-500/10 border-yellow-500/30",
  low:      "text-blue-400 bg-blue-400/10 border-blue-400/30",
};

function HighRiskFunctions({ r }: { r: FullAnalysisResult }) {
  const fr = r.function_risk;
  if (!fr || !fr.localization_available || fr.total_functions === 0) return null;

  const top = fr.top_k.slice(0, 3);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-foreground">High Risk Functions</h3>
        <span className="text-xs text-muted-foreground">
          {fr.n_high_risk} of {fr.total_functions} functions flagged
        </span>
      </div>
      {top.length === 0 ? (
        <p className="text-sm text-muted-foreground">No high-risk functions detected.</p>
      ) : (
        <div className="space-y-2">
          {top.map((fn) => (
            <div
              key={fn.name}
              className={`rounded-md border p-3 ${riskColors[fn.risk_level] ?? riskColors.low}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-mono text-sm font-semibold truncate">{fn.name}()</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded border font-medium shrink-0 ${riskColors[fn.risk_level]}`}>
                    {fn.risk_level}
                  </span>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                  <MapPin className="w-3 h-3" />
                  <span>L{fn.lineno}–{fn.end_lineno}</span>
                </div>
              </div>
              <p className="text-xs mt-1.5 opacity-80">{fn.reason}</p>
              <div className="flex gap-4 mt-2 text-xs opacity-70">
                <span>Cognitive: {fn.cognitive_complexity}</span>
                <span>Cyclomatic: {fn.cyclomatic_complexity}</span>
                <span>{fn.sloc} lines</span>
                <span>Risk share: {Math.round(fn.complexity_weight * 100)}%</span>
              </div>
            </div>
          ))}
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
        <div key={`${clone.start_line_a}-${clone.start_line_b}-${i}`} className="bg-secondary/30 rounded-xl p-4 space-y-2">
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
        <div key={s.title ?? i} className={`bg-secondary/30 rounded-xl p-4 space-y-2 border-l-4 ${SEV_BORDER[s.priority] ?? "border-l-border"}`}>
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
        <div key={issue.title ?? i} className={`bg-secondary/30 rounded-xl p-4 space-y-1 border-l-4 ${SEV_BORDER[issue.severity] ?? "border-l-border"}`}>
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
      {d.symbol_scores.length > 15 && (
        <p className="text-xs text-muted-foreground mb-2">Showing 15 of {d.symbol_scores.length} symbols</p>
      )}
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
              <tr key={s.name ?? i} className="border-b border-border/40">
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
        <div key={issue.title ?? i} className={`bg-secondary/30 rounded-xl p-4 space-y-2 border-l-4 ${SEV_BORDER[issue.severity] ?? "border-l-border"}`}>
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
            <div key={issue.title ?? i} className="bg-secondary/20 rounded-lg p-3 mb-2 space-y-1">
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
              <li key={`tip-${i}`} className="flex items-start gap-2 text-sm text-muted-foreground">
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

const AST_FEATURE_META: Record<string, { label: string; unit: string; warnAbove?: number; warnBelow?: number }> = {
  n_functions:              { label: "Functions",           unit: "",    warnAbove: 10 },
  n_classes:                { label: "Classes",             unit: "",    warnAbove: 5 },
  n_try_blocks:             { label: "Try blocks",          unit: "",    warnAbove: 8 },
  n_raises:                 { label: "Raises / throws",     unit: "",    warnAbove: 10 },
  n_with_blocks:            { label: "With-statements",     unit: "",    warnAbove: 8 },
  max_nesting_depth:        { label: "Max nesting depth",   unit: "",    warnAbove: 4 },
  max_params:               { label: "Max parameters",      unit: "",    warnAbove: 5 },
  avg_params:               { label: "Avg parameters",      unit: "",    warnAbove: 3.5 },
  n_decorated_functions:    { label: "Decorated functions", unit: "",    warnAbove: 6 },
  n_imports:                { label: "Imports",             unit: "",    warnAbove: 15 },
  max_function_body_lines:  { label: "Max function length", unit: " ln", warnAbove: 50 },
  avg_function_body_lines:  { label: "Avg function length", unit: " ln", warnAbove: 25 },
};

function PatternTab({ r }: { r: FullAnalysisResult }) {
  const p = r.patterns;
  if (!p) return <Empty msg="Pattern analysis not available." />;
  const criticalSecFindings = r.security?.summary?.critical ?? 0;
  const astFeatures = p.raw_ast_features ?? {};

  const hasAst = Object.keys(astFeatures).length > 0;

  return (
    <div className="space-y-6">
      {criticalSecFindings > 0 && p.label === "clean" && (
        <div className="flex items-start gap-3 bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
          <AlertTriangle className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" />
          <p className="text-sm text-orange-300">
            <strong>Note:</strong> The pattern model evaluates code structure only — not security semantics.
            This file has <strong>{criticalSecFindings} critical security finding{criticalSecFindings !== 1 ? "s" : ""}</strong>; see the Security tab.
          </p>
        </div>
      )}

      <div className="flex items-start gap-3 bg-secondary/20 border border-border rounded-lg p-3">
        <Info className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
        <p className="text-sm text-muted-foreground">
          Structural quality scores extracted directly from the AST. No classifier label is shown because
          inter-rater agreement on four-class pattern labels is low (kappa = 0.17); raw feature values
          are more actionable than a predicted category.
        </p>
      </div>

      {hasAst ? (
        <div>
          <p className="text-sm font-semibold text-foreground mb-3">Structural AST Features</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(AST_FEATURE_META)
              .filter(([key]) => key in astFeatures)
              .map(([key, meta]) => {
                const val = astFeatures[key] ?? 0;
                const warn = (meta.warnAbove !== undefined && val > meta.warnAbove) ||
                             (meta.warnBelow !== undefined && val < meta.warnBelow);
                return (
                  <div
                    key={key}
                    className={`rounded-lg p-3 border ${warn ? "border-orange-500/40 bg-orange-500/5" : "border-border bg-secondary/20"}`}
                  >
                    <p className="text-xs text-muted-foreground mb-1">{meta.label}</p>
                    <p className={`text-xl font-bold ${warn ? "text-orange-400" : "text-foreground"}`}>
                      {typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(1)) : val}
                      <span className="text-sm font-normal text-muted-foreground">{meta.unit}</span>
                    </p>
                    {warn && (
                      <p className="text-xs text-orange-400 mt-1">
                        Above recommended threshold ({meta.warnAbove}{meta.unit})
                      </p>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      ) : (
        <div className="text-sm text-muted-foreground italic">
          AST feature scores not available for this analysis. Re-analyse to get raw scores.
        </div>
      )}

      <div className="bg-secondary/20 rounded-xl p-4 flex items-center gap-3">
        <CheckCircle className="w-4 h-4 text-muted-foreground shrink-0" />
        <p className="text-sm text-muted-foreground">
          12 structural features extracted from AST node counts and control-flow depth.
          Thresholds reflect empirical distributions from the training corpus.
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

// ─── Complexity fix templates (SHAP-guided: keyed by primary metric driver) ──

const COMPLEXITY_TEMPLATES: Record<string, { title: string; before: string; after: string; explanation: string }> = {
  high_cyclomatic: {
    title: "Reduce Cyclomatic Complexity",
    before: "def process(x):\n    if x > 0:\n        if x > 10:\n            if x > 100:\n                return 'large'\n            return 'medium'\n        return 'small'\n    return 'negative'",
    after: "THRESHOLDS = [(100, 'large'), (10, 'medium'), (0, 'small')]\n\ndef classify(x):\n    for threshold, label in THRESHOLDS:\n        if x > threshold:\n            return label\n    return 'negative'\n\ndef process(x):\n    return classify(x)",
    explanation: "Cyclomatic complexity is elevated. Replace deep if/else chains with early returns, lookup tables, or extracted helper functions. Target CC <= 10 per function.",
  },
  long_functions: {
    title: "Break Up Long Functions",
    before: "def do_everything(data, config):\n    # validation\n    if not data: raise ValueError(...)\n    # processing — 60+ more lines\n    ...",
    after: "def _validate(data): ...\ndef _process(data, config): ...\ndef _format(result): ...\n\ndef do_everything(data, config):\n    _validate(data)\n    result = _process(data, config)\n    return _format(result)",
    explanation: "Long functions (> 50 lines) are a primary complexity driver. Decompose into single-responsibility helpers. Each function should fit on one screen.",
  },
  complex_functions: {
    title: "Extract Complex Function Logic",
    before: "def render(data, config, user, fmt, opts):\n    # 15+ branching paths, deeply nested\n    ...",
    after: "def _select_template(fmt, opts): ...\ndef _apply_permissions(data, user): ...\n\ndef render(data, config, user, fmt, opts):\n    tpl = _select_template(fmt, opts)\n    data = _apply_permissions(data, user)\n    return tpl.render(data, config)",
    explanation: "Functions with CC > 10 are the top driver of your complexity score. Extract discrete logical sections into named helpers — this directly reduces the complexity model's prediction.",
  },
  high_halstead: {
    title: "Simplify Dense Operator Usage",
    before: "result = (a + b * c - d / e) if (x > y and z != w or p <= q) else (f * g + h)",
    after: "numerator = a + b * c - d / e\ncondition = x > y and (z != w or p <= q)\nfallback = f * g + h\nresult = numerator if condition else fallback",
    explanation: "High Halstead volume/difficulty indicates dense operator and operand usage. Introduce intermediate named variables to clarify intent and reduce estimated bug density.",
  },
  lines_over_80: {
    title: "Fix Long Lines",
    before: "result = some_function(arg_one, arg_two, arg_three, key_one=val_one, key_two=val_two)",
    after: "result = some_function(\n    arg_one,\n    arg_two,\n    arg_three,\n    key_one=val_one,\n    key_two=val_two,\n)",
    explanation: "Many lines exceed 80 characters, reducing readability scores. Break long calls and expressions using Python's implicit line continuation inside parentheses.",
  },
  high_cognitive: {
    title: "Reduce Cognitive Complexity",
    before: "for item in items:\n    if item.active:\n        if item.type == 'A':\n            for sub in item.children:\n                if sub.valid:\n                    process(sub)  # nesting depth 4",
    after: "def _process_valid_children(item):\n    for sub in item.children:\n        if sub.valid:\n            process(sub)\n\nfor item in items:\n    if item.active and item.type == 'A':\n        _process_valid_children(item)",
    explanation: "Cognitive complexity measures how hard code is to understand. Flatten nesting by inverting conditions, using early returns, and extracting inner loops into helpers.",
  },
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

  // ── Complexity fixes — SHAP-guided via metric elevation as importance proxy ─
  // Rank each metric by how far it is above its acceptable threshold.
  // The metric with the highest elevation score is the top driver of the
  // complexity prediction (mirrors SHAP feature attribution logic).
  const cx = r.complexity;
  if (cx && (cx.grade === "C" || cx.grade === "D" || cx.grade === "F")) {
    const drivers: Array<{ key: string; elevationScore: number }> = [];
    if (cx.cyclomatic > 10)         drivers.push({ key: "high_cyclomatic",  elevationScore: cx.cyclomatic / 10 });
    if (cx.cognitive > 15)          drivers.push({ key: "high_cognitive",   elevationScore: cx.cognitive / 15 });
    if (cx.n_complex_functions > 0) drivers.push({ key: "complex_functions",elevationScore: cx.n_complex_functions * 1.5 });
    if (cx.n_long_functions > 0)    drivers.push({ key: "long_functions",   elevationScore: cx.n_long_functions * 1.2 });
    if (cx.halstead_bugs > 0.5)     drivers.push({ key: "high_halstead",    elevationScore: cx.halstead_bugs * 8 });
    if (cx.n_lines_over_80 > 5)     drivers.push({ key: "lines_over_80",    elevationScore: cx.n_lines_over_80 / 5 });

    // Sort descending — highest elevation = primary driver = highest priority fix
    drivers.sort((a, b) => b.elevationScore - a.elevationScore);
    const cxPriority: "high" | "medium" = cx.grade === "D" || cx.grade === "F" ? "high" : "medium";

    drivers.slice(0, 3).forEach(({ key }) => {
      const tpl = COMPLEXITY_TEMPLATES[key];
      if (tpl) {
        fixes.push({
          priority: cxPriority,
          category: "Complexity",
          title: tpl.title,
          before: tpl.before,
          after: tpl.after,
          explanation: tpl.explanation,
        });
      }
    });
  }

  // ── Performance fixes ───────────────────────────────────────────────────────
  (r.performance?.issues ?? []).slice(0, 3).forEach((p) => {
    fixes.push({
      priority: p.severity as "high" | "medium" | "low",
      category: "Performance",
      title: p.title,
      explanation: p.speedup_hint ? `${p.description}\n\nHint: ${p.speedup_hint}` : p.description,
    });
  });

  // ── Documentation fixes ─────────────────────────────────────────────────────
  const docCoverage = r.docs?.coverage ?? 1;
  if (r.docs && docCoverage < 0.7) {
    const undocumented = r.docs.symbol_scores.filter((s) => !s.has_docstring).slice(0, 2).map((s) => s.name);
    fixes.push({
      priority: docCoverage < 0.3 ? "high" : "medium",
      category: "Documentation",
      title: `Add docstrings to ${r.docs.total_symbols - r.docs.documented_symbols} undocumented symbol${r.docs.total_symbols - r.docs.documented_symbols !== 1 ? "s" : ""}`,
      before: undocumented.length > 0 ? `def ${undocumented[0]}(...):\n    pass  # no docstring` : undefined,
      after: undocumented.length > 0
        ? `def ${undocumented[0]}(...):\n    """Brief description.\n\n    Args:\n        ...\n\n    Returns:\n        ...\n    """\n    pass`
        : undefined,
      explanation: `Documentation coverage is ${Math.round(docCoverage * 100)}% (grade: ${r.docs.grade}). Add docstrings to all public functions, classes, and modules.`,
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

  // Derive the top complexity driver for the insight banner
  const cxInsight = (() => {
    if (!cx || (cx.grade !== "C" && cx.grade !== "D" && cx.grade !== "F")) return null;
    const candidates: Array<{ label: string; value: string; score: number }> = [];
    if (cx.cyclomatic > 10)         candidates.push({ label: "cyclomatic complexity",      value: `${cx.cyclomatic}`,         score: cx.cyclomatic / 10 });
    if (cx.cognitive > 15)          candidates.push({ label: "cognitive complexity",        value: `${cx.cognitive}`,          score: cx.cognitive / 15 });
    if (cx.n_complex_functions > 0) candidates.push({ label: "functions with CC > 10",     value: `${cx.n_complex_functions}`, score: cx.n_complex_functions * 1.5 });
    if (cx.n_long_functions > 0)    candidates.push({ label: "functions over 50 lines",    value: `${cx.n_long_functions}`,   score: cx.n_long_functions * 1.2 });
    if (cx.halstead_bugs > 0.5)     candidates.push({ label: "estimated bug density",      value: `${cx.halstead_bugs.toFixed(2)}`, score: cx.halstead_bugs * 8 });
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0] ?? null;
  })();

  return (
    <div className="space-y-4">
      {cxInsight && (
        <div className="flex items-start gap-3 px-4 py-3 bg-orange-500/8 border border-orange-500/25 rounded-xl">
          <span className="text-orange-400 mt-0.5 shrink-0 text-base">~</span>
          <p className="text-sm text-orange-300">
            <span className="font-semibold">Primary complexity driver:</span>{" "}
            {cxInsight.label} = <span className="font-mono font-semibold">{cxInsight.value}</span>.{" "}
            The fixes below are ordered by their expected impact on your complexity score (grade: {cx!.grade}).
          </p>
        </div>
      )}
      <p className="text-sm text-muted-foreground">{sorted.length} fix recommendation{sorted.length !== 1 ? "s" : ""} — sorted by priority</p>
      {sorted.map((fix, i) => (
        <div key={fix.title ?? i} className="bg-card border border-border rounded-xl overflow-hidden">
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

// ─── ExplainTab ───────────────────────────────────────────────────────────────

function ExplainTab({ r, repoName }: { r: FullAnalysisResult; repoName?: string }) {
  const [data, setData] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (!r.filename) return;
    let code = (r as unknown as Record<string, string>)._source ?? r.code ?? "";
    // If source not stored (batch scan), try fetching from GitHub
    if (!code && repoName) {
      try {
        code = await getRawFile(repoName, "main", r.filename);
      } catch {
        try {
          code = await getRawFile(repoName, "master", r.filename);
        } catch { /* fall through */ }
      }
    }
    if (!code) {
      setError("Source code not available for explanation (stored result may be truncated).");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await explainAnalysis(code, r.filename);
      setData(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const IMPACT_CLS: Record<string, string> = {
    high:   "bg-red-500/20 text-red-400 border border-red-500/30",
    medium: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
    low:    "bg-blue-500/20 text-blue-400 border border-blue-500/30",
  };

  if (!data && !loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center gap-4">
        <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
          <Brain className="w-7 h-7 text-primary" />
        </div>
        <div>
          <p className="text-lg font-semibold text-foreground">ML Explanation</p>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">
            Understand exactly which code features drove each model's prediction — feature importance, risk drivers, and actionable recommendations.
          </p>
        </div>
        {error && <p className="text-sm text-destructive max-w-sm">{error}</p>}
        <button
          onClick={load}
          disabled={loading}
          className="px-5 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? "Generating..." : "Generate Explanation"}
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-muted-foreground">Generating ML explanations...</p>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Actionable summary */}
      <div className="bg-primary/5 border border-primary/20 rounded-xl p-5">
        <p className="text-xs font-semibold text-primary uppercase tracking-wide mb-3">Actionable Summary</p>
        <ul className="space-y-2">
          {data.actionable_summary.map((item, i) => (
            <li key={`as-${i}`} className="flex items-start gap-2 text-sm text-foreground">
              <span className="text-primary font-bold shrink-0">{i + 1}.</span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Complexity model */}
      {data.models.complexity && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-sm font-semibold text-foreground mb-1">Complexity Model (XGBoost)</p>
          <p className="text-xs text-muted-foreground mb-4">{data.models.complexity.interpretation}</p>
          {data.models.complexity.top_features.length > 6 && (
            <p className="text-xs text-muted-foreground mb-2">Showing top 6 of {data.models.complexity.top_features.length} features</p>
          )}
          <div className="space-y-2">
            {data.models.complexity.top_features.slice(0, 6).map((f) => (
              <div key={f.feature} className="flex items-center gap-3">
                <span className="text-xs font-mono text-foreground w-44 truncate shrink-0">{f.feature}</span>
                <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full"
                    style={{ width: `${Math.min(100, f.importance * 400)}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-12 text-right shrink-0">
                  {f.value}
                </span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold shrink-0 ${IMPACT_CLS[f.impact] ?? ""}`}>
                  {f.impact}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Security model */}
      {data.models.security && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-sm font-semibold text-foreground mb-1">Security Model (RF + CNN)</p>
          <p className="text-xs text-muted-foreground mb-4">{data.models.security.interpretation}</p>
          {(() => {
            const allFeats = Object.entries(data.models.security!.rf_features).filter(([, v]) => v > 0);
            if (allFeats.length > 9) return <p className="text-xs text-muted-foreground mb-2">Showing top 9 of {allFeats.length} features</p>;
            return null;
          })()}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(data.models.security.rf_features)
              .filter(([, v]) => v > 0)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 9)
              .map(([k, v]) => (
                <div key={k} className={`flex justify-between p-2 rounded-lg text-xs border ${
                  data.models.security!.high_risk_signals.includes(k)
                    ? "bg-red-500/10 border-red-500/30 text-red-400"
                    : "bg-secondary/30 border-border text-muted-foreground"
                }`}>
                  <span className="font-mono truncate">{k.replace("n_", "")}</span>
                  <span className="font-bold ml-2 shrink-0">{v}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Bug risk model */}
      {data.models.bug && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-sm font-semibold text-foreground mb-1">Bug Risk Model (XGBoost + LR)</p>
          <p className="text-xs text-muted-foreground mb-4">{data.models.bug.interpretation}</p>
          <div className="space-y-2">
            {data.models.bug.risk_drivers.map((d) => (
              <div key={d.feature} className={`flex items-center gap-3 p-2.5 rounded-lg border text-xs ${
                d.exceeded ? "bg-orange-500/10 border-orange-500/30" : "bg-secondary/20 border-border"
              }`}>
                <span className={`font-mono flex-1 truncate ${d.exceeded ? "text-orange-300" : "text-muted-foreground"}`}>
                  {d.feature}
                </span>
                <span className="text-foreground font-semibold shrink-0">{d.value}</span>
                <span className="text-muted-foreground shrink-0">/ {d.threshold}</span>
                {d.exceeded && <span className="text-orange-400 font-bold shrink-0">!</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={load}
        className="text-xs text-muted-foreground hover:text-foreground underline transition-colors"
      >
        Refresh explanations
      </button>
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
  { id: "explain",     label: "Explain",       Icon: Brain },
];

const ReviewDetail = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams<{ id: string }>();

  // Try: router state → history by ID → null (show mock)
  const entry = id ? getEntry(id) : null;
  const result: FullAnalysisResult | null =
    location.state?.result ?? (entry?.result ?? null);
  const entryRepoName: string | undefined = entry?.repoName ?? location.state?.repoName;

  // ── Create PR state ──
  const [prModal, setPrModal] = useState(false);
  const [prRepos, setPrRepos] = useState<GitHubRepo[]>([]);
  const [prRepo, setPrRepo] = useState("");
  const [prBranch, setPrBranch] = useState("");
  const [prTitle, setPrTitle] = useState("");
  const [prLoading, setPrLoading] = useState(false);
  const [prResult, setPrResult] = useState<{ pr_url: string; comment_url: string; issue_urls: string[] } | null>(null);

  useEffect(() => {
    if (prModal && prRepos.length === 0 && isGitHubConnected()) {
      listUserRepos(1, 50).then(setPrRepos).catch(() => {});
    }
  }, [prModal]);

  const handleCreatePR = async () => {
    if (!result || !prRepo) return;
    const token = getGitHubToken();
    if (!token) { toast.error("GitHub not connected"); return; }

    setPrLoading(true);
    try {
      // Build a short markdown summary of findings
      const findings = result.security?.findings ?? [];
      const bugLevel = result.bug_prediction?.risk_level ?? "unknown";
      const score = result.overall_score ?? 0;
      let summary = `### Analysis Summary\n\n`;
      summary += `- Overall score: **${score}/100**\n`;
      summary += `- Bug risk: **${bugLevel}**\n`;
      summary += `- Security findings: **${findings.length}**\n`;
      if (findings.length > 0) {
        summary += `\n#### Security Findings\n`;
        findings.slice(0, 10).forEach((f: any) => {
          summary += `- **${f.severity?.toUpperCase()}** L${f.lineno}: ${f.title} — ${f.description}\n`;
        });
      }
      summary += `\n_Generated by [IntelliCode](https://intellcode.onrender.com)_`;

      const res = await fetch(`${BASE_URL}/github/create-pr`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({
          repo_full_name: prRepo,
          filename: result.filename || "reviewed_code.py",
          code: result.code ?? "",
          branch_name: prBranch.trim() || "",
          pr_title: prTitle.trim() || `IntelliCode Review: ${result.filename}`,
          analysis_summary: summary,
          security_findings: findings,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setPrResult({ pr_url: data.pr_url, comment_url: data.comment_url, issue_urls: data.issue_urls ?? [] });
      const issueCount = (data.issue_urls ?? []).length;
      toast.success(`PR created on GitHub!${issueCount ? ` ${issueCount} critical issue${issueCount > 1 ? "s" : ""} opened.` : ""}`);
    } catch (e: any) {
      toast.error(e.message ?? "Failed to create PR");
    } finally {
      setPrLoading(false);
    }
  };

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
            {isGitHubConnected() && result.code ? (
              <Button variant="outline" size="sm" onClick={() => { setPrModal(true); setPrResult(null); }}>
                <Github className="w-4 h-4 mr-1" />
                Create PR
              </Button>
            ) : isGitHubConnected() && (
              <Button variant="outline" size="sm" disabled title="Re-analyze this file to enable PR creation">
                <Github className="w-4 h-4 mr-1" />
                Create PR
              </Button>
            )}
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
            <TabsContent value="explain">    <ExplainTab r={result} repoName={entryRepoName} /></TabsContent>
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

      {/* Create PR Modal */}
      {prModal && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setPrModal(false)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-md w-full space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-foreground flex items-center gap-2">
                <Github className="w-4 h-4" /> Create GitHub PR
              </h3>
              <button onClick={() => setPrModal(false)} className="text-muted-foreground hover:text-foreground">
                <X className="w-5 h-5" />
              </button>
            </div>

            {prResult ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-green-500 text-sm font-medium">
                  <CheckCircle className="w-4 h-4" /> PR created successfully!
                </div>
                <a href={prResult.pr_url} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-primary hover:underline">
                  <ExternalLink className="w-3.5 h-3.5" /> View Pull Request
                </a>
                {prResult.issue_urls.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-destructive">
                      {prResult.issue_urls.length} critical finding{prResult.issue_urls.length > 1 ? "s" : ""} opened as GitHub Issues:
                    </p>
                    {prResult.issue_urls.map((url, i) => (
                      <a key={url} href={url} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground hover:underline">
                        <ExternalLink className="w-3 h-3" /> Issue #{i + 1}
                      </a>
                    ))}
                  </div>
                )}
                <Button size="sm" variant="outline" className="w-full" onClick={() => setPrModal(false)}>
                  Close
                </Button>
              </div>
            ) : (
              <>
                <div className="space-y-1">
                  <label className="text-sm font-medium text-foreground">Repository</label>
                  <select
                    value={prRepo}
                    onChange={(e) => setPrRepo(e.target.value)}
                    className="w-full rounded-lg bg-input border border-border text-sm text-foreground px-3 py-2 focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="">Select a repo...</option>
                    {prRepos.map((r) => (
                      <option key={r.id} value={r.full_name}>{r.full_name}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-medium text-foreground">Branch name <span className="text-muted-foreground font-normal">(optional)</span></label>
                  <input
                    value={prBranch}
                    onChange={(e) => setPrBranch(e.target.value)}
                    placeholder="intellcode/review-..."
                    className="w-full rounded-lg bg-input border border-border text-sm text-foreground px-3 py-2 focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-sm font-medium text-foreground">PR title <span className="text-muted-foreground font-normal">(optional)</span></label>
                  <input
                    value={prTitle}
                    onChange={(e) => setPrTitle(e.target.value)}
                    placeholder={`IntelliCode Review: ${result?.filename ?? "code"}`}
                    className="w-full rounded-lg bg-input border border-border text-sm text-foreground px-3 py-2 focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>

                <p className="text-xs text-muted-foreground">
                  This will commit the analyzed code to a new branch in the selected repo and open a PR with findings as a comment.
                </p>

                <div className="flex gap-3 justify-end">
                  <Button variant="outline" size="sm" onClick={() => setPrModal(false)}>Cancel</Button>
                  <Button
                    size="sm"
                    className="bg-gradient-primary"
                    onClick={handleCreatePR}
                    disabled={!prRepo || prLoading}
                  >
                    {prLoading ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />Creating...</> : <><Github className="w-3.5 h-3.5 mr-1.5" />Create PR</>}
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewDetail;
