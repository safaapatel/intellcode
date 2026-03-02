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
  Download,
} from "lucide-react";
import { mockReviewResult, mockIssues } from "@/data/mockData";
import { getEntry } from "@/services/reviewHistory";
import { toast } from "sonner";
import type { FullAnalysisResult } from "@/types/analysis";

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

function ScoreCircle({ score, label, size = 128 }: { score: number; label: string; size?: number }) {
  const r = size / 2 - 8;
  const circ = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, score));
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="w-full h-full -rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none"
            className={scoreStroke(clamped)} strokeWidth="8"
            strokeDasharray={`${(clamped / 100) * circ} ${circ}`} strokeLinecap="round" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${size >= 100 ? "text-3xl" : "text-xl"} ${scoreText(clamped)}`}>
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
            value: r.security.summary.total,
            sub: `${r.security.summary.critical} critical`,
            cls: r.security.summary.critical > 0 ? "text-destructive" : "text-foreground",
          },
          {
            label: "Bug Risk",
            value: r.bug_prediction.risk_level.toUpperCase(),
            sub: `${Math.round(r.bug_prediction.bug_probability * 100)}% probability`,
            cls: ({ low: "text-green-500", medium: "text-yellow-500", high: "text-orange-500", critical: "text-destructive" } as Record<string, string>)[r.bug_prediction.risk_level] ?? "text-foreground",
          },
          {
            label: "Clones Found",
            value: r.clones.clones.length,
            sub: `${(r.clones.clone_rate * 100).toFixed(0)}% duplication`,
            cls: "text-foreground",
          },
          {
            label: "Dead Code",
            value: `${r.dead_code.dead_line_count} lines`,
            sub: `${(r.dead_code.dead_ratio * 100).toFixed(1)}% of file`,
            cls: "text-foreground",
          },
        ].map((card) => (
          <div key={card.label} className="bg-secondary/40 rounded-xl p-4 text-center">
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

function SecurityTab({ r }: { r: FullAnalysisResult }) {
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
      {findings.map((f, i) => (
        <div key={i} className="bg-secondary/30 rounded-xl p-4 space-y-2">
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
          <p className="text-xs text-muted-foreground">Source: {f.source} · Confidence: {Math.round(f.confidence * 100)}%</p>
        </div>
      ))}
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
        <ScoreCircle score={pct} label="Bug Probability %" />
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
        <div key={i} className="bg-secondary/30 rounded-xl p-4 space-y-2">
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
        <div key={i} className="bg-secondary/30 rounded-xl p-4 space-y-1">
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
        <div key={i} className="bg-secondary/30 rounded-xl p-4 space-y-2">
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

// ─── Page ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: "overview",     label: "Overview",      Icon: Eye },
  { id: "security",    label: "Security",      Icon: Shield },
  { id: "complexity",  label: "Complexity",    Icon: Gauge },
  { id: "bugs",        label: "Bug Risk",      Icon: Bug },
  { id: "clones",      label: "Clones",        Icon: Copy },
  { id: "refactoring", label: "Refactoring",   Icon: Wrench },
  { id: "deadcode",    label: "Dead Code",     Icon: FileX },
  { id: "debt",        label: "Debt",          Icon: CreditCard },
  { id: "docs",        label: "Docs",          Icon: BookOpen },
  { id: "performance", label: "Performance",   Icon: Zap },
  { id: "deps",        label: "Dependencies",  Icon: Network },
  { id: "readability", label: "Readability",   Icon: Eye },
];

const ReviewDetail = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams<{ id: string }>();

  // Try: router state → history by ID → null (show mock)
  const result: FullAnalysisResult | null =
    location.state?.result ?? (id ? getEntry(id)?.result ?? null : null);

  const handleExport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `intellicode-report-${Date.now()}.json`,
    });
    a.click();
    toast.success("Report exported as JSON");
  };

  // ── Fallback: no real result (e.g. /reviews/1 accessed directly) ──
  if (!result) {
    return (
      <div className="min-h-screen bg-background">
        <AppNavigation />
        <main className="container mx-auto px-4 py-8 max-w-5xl">
          <div className="bg-card border border-border rounded-xl p-8 mb-6">
            <div className="flex flex-col items-center mb-6">
              <div className="relative w-32 h-32 mb-4">
                <svg className="w-full h-full -rotate-90">
                  <circle cx="64" cy="64" r="56" fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
                  <circle cx="64" cy="64" r="56" fill="none" className="stroke-primary" strokeWidth="8"
                    strokeDasharray={`${(mockReviewResult.overallScore / 100) * 352} 352`} strokeLinecap="round" />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-foreground">{mockReviewResult.overallScore}</span>
                  <span className="text-xs text-muted-foreground">Code Quality</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground text-center">
                This is sample data.{" "}
                <span className="text-primary cursor-pointer underline" onClick={() => navigate("/submit")}>
                  Submit real code
                </span>{" "}
                to see live ML results across 12 models.
              </p>
            </div>
            <div className="space-y-3">
              {mockIssues.map((issue) => (
                <div key={issue.id} className="bg-secondary/30 rounded-xl p-4 space-y-1">
                  <div className="flex items-center gap-2">
                    <SevBadge sev={issue.severity} />
                    <span className="font-semibold text-sm text-foreground">{issue.title}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{issue.description}</p>
                </div>
              ))}
            </div>
          </div>
          <Button variant="outline" onClick={() => navigate("/dashboard")}>← Back</Button>
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
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="text-sm text-muted-foreground mb-1">
              <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>
                Dashboard
              </span>
              <span className="mx-2">›</span>
              <span className="text-foreground">{result.filename}</span>
            </div>
            <h1 className="text-2xl font-bold text-foreground">Analysis Report</h1>
            <p className="text-sm text-muted-foreground mt-1">
              {result.language} · {result.duration_seconds}s · 12 models
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="w-4 h-4 mr-2" />
            Export JSON
          </Button>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview">
          <TabsList className="flex flex-wrap h-auto gap-1 mb-6 bg-secondary/40 p-1 rounded-xl">
            {TABS.map(({ id, label, Icon }) => (
              <TabsTrigger
                key={id} value={id}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 data-[state=active]:bg-background data-[state=active]:text-foreground rounded-lg"
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </TabsTrigger>
            ))}
          </TabsList>

          <div className="bg-card border border-border rounded-xl p-6 min-h-[300px]">
            <TabsContent value="overview">    <OverviewTab r={result} /></TabsContent>
            <TabsContent value="security">   <SecurityTab r={result} /></TabsContent>
            <TabsContent value="complexity"> <ComplexityTab r={result} /></TabsContent>
            <TabsContent value="bugs">       <BugTab r={result} /></TabsContent>
            <TabsContent value="clones">     <ClonesTab r={result} /></TabsContent>
            <TabsContent value="refactoring"><RefactoringTab r={result} /></TabsContent>
            <TabsContent value="deadcode">   <DeadCodeTab r={result} /></TabsContent>
            <TabsContent value="debt">       <DebtTab r={result} /></TabsContent>
            <TabsContent value="docs">       <DocsTab r={result} /></TabsContent>
            <TabsContent value="performance"><PerformanceTab r={result} /></TabsContent>
            <TabsContent value="deps">       <DepsTab r={result} /></TabsContent>
            <TabsContent value="readability"><ReadabilityTab r={result} /></TabsContent>
          </div>
        </Tabs>

        <div className="flex items-center justify-between mt-6">
          <Button variant="outline" onClick={() => navigate("/dashboard")}>← Back to Dashboard</Button>
          <Button className="bg-gradient-primary" onClick={() => navigate("/submit")}>
            Analyze Another File
          </Button>
        </div>
      </main>
    </div>
  );
};

export default ReviewDetail;
