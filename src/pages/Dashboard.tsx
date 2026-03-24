import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { QuickActions } from "@/components/app/QuickActions";
import { Button } from "@/components/ui/button";
import { mockUser } from "@/data/mockData";
import { getSession } from "@/services/auth";
import { getEntries, getDashboardStats } from "@/services/reviewHistory";
import {
  ExternalLink, Clock, FileCode2, AlertTriangle, ClipboardCheck,
  CheckCircle2, TrendingUp, Shield, Zap, BarChart3, Plus, GitCompare,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

function savedFirstName(): string {
  try {
    const raw = localStorage.getItem("intellcode_settings");
    const name = raw ? JSON.parse(raw)?.name : null;
    return name ? name.split(" ")[0] : mockUser.name.split(" ")[0];
  } catch {
    return mockUser.name.split(" ")[0];
  }
}

const SEV_CONFIG: Record<string, { cls: string; bar: string }> = {
  critical: { cls: "bg-red-600 text-white",    bar: "bg-red-500" },
  high:     { cls: "bg-orange-500 text-white",  bar: "bg-orange-400" },
  medium:   { cls: "bg-yellow-500 text-black",  bar: "bg-yellow-400" },
  low:      { cls: "bg-blue-500 text-white",    bar: "bg-blue-400" },
  none:     { cls: "bg-emerald-600 text-white", bar: "bg-emerald-500" },
};

const STATUS_CLS: Record<string, string> = {
  completed:   "text-emerald-400",
  in_progress: "text-yellow-400",
  pending:     "text-blue-400",
};

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function ScoreMini({ score }: { score: number }) {
  const color = score >= 80 ? "bg-emerald-500" : score >= 60 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 shrink-0">
      <div className="w-16 h-1.5 bg-secondary rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs font-semibold tabular-nums text-foreground w-6 text-right">{score}</span>
    </div>
  );
}

function loadFeedbackKeys(): Set<string> {
  try {
    const raw = localStorage.getItem("intellcode_feedback");
    if (!raw) return new Set();
    const all = JSON.parse(raw);
    const keys = new Set<string>();
    Object.values(all).forEach((ff) => {
      Object.keys(ff as Record<string, unknown>).forEach((k) => keys.add(k));
    });
    return keys;
  } catch {
    return new Set();
  }
}

const Dashboard = () => {
  const navigate = useNavigate();
  const session = getSession();
  const isReviewer = session?.role === "reviewer";
  const [firstName, setFirstName] = useState(savedFirstName);
  const [historyVersion, setHistoryVersion] = useState(0);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "intellcode_settings") setFirstName(savedFirstName());
      if (e.key === "intellcode_review_history") setHistoryVersion((v) => v + 1);
    };
    window.addEventListener("storage", onStorage);
    const id = setInterval(() => {
      setFirstName(savedFirstName());
      setHistoryVersion((v) => v + 1);
    }, 5000);
    return () => { window.removeEventListener("storage", onStorage); clearInterval(id); };
  }, []);

  const realStats = historyVersion >= 0 ? getDashboardStats() : null;
  const allEntries = historyVersion >= 0 ? getEntries() : [];
  const recentEntries = allEntries.slice(0, 5);
  // Dedup by filename — keep the latest score per file
  const latestByFile = useMemo(() => {
    const seen = new Map<string, typeof allEntries[number]>();
    allEntries.forEach((e) => { if (!seen.has(e.filename)) seen.set(e.filename, e); });
    return [...seen.values()];
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyVersion]);
  const topFiles  = [...latestByFile].sort((a, b) => b.overallScore - a.overallScore).slice(0, 5);
  const worstFiles = [...latestByFile].sort((a, b) => a.overallScore - b.overallScore).filter(f => f.overallScore < 80).slice(0, 5);
  const trendData = useMemo(
    () =>
      [...getEntries()]
        .reverse()
        .slice(-20)
        .map((e) => ({
          date: new Date(e.submittedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
          score: e.overallScore,
        })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [historyVersion]
  );

  const pendingReviewEntries = isReviewer
    ? getEntries().filter((e) => {
        const findings = e.result?.security?.findings ?? [];
        if (findings.length === 0) return false;
        const fbKeys = loadFeedbackKeys();
        return findings.some(
          (f: { title: string; lineno: number }) => !fbKeys.has(`${f.title}-${f.lineno}`)
        );
      })
    : [];

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">

        {/* ── Hero Banner ── */}
        <div className="relative overflow-hidden rounded-2xl mb-8 border border-primary/20 bg-gradient-to-br from-primary/15 via-primary/8 to-transparent">
          <div className="absolute -top-16 -right-16 w-56 h-56 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 left-1/2 w-40 h-20 bg-primary/5 rounded-full blur-2xl pointer-events-none" />
          <div className="relative z-10 flex items-center justify-between p-6 sm:p-8">
            <div>
              <p className="text-sm text-primary font-semibold mb-1 tracking-wide">{greeting} 👋</p>
              <h1 className="text-2xl font-bold text-foreground mb-1">Welcome back, {firstName}!</h1>
              <p className="text-sm text-muted-foreground">
                {isReviewer
                  ? `${pendingReviewEntries.length} submission${pendingReviewEntries.length !== 1 ? "s" : ""} awaiting your review`
                  : recentEntries.length > 0
                  ? `${recentEntries.length} review${recentEntries.length !== 1 ? "s" : ""} in history · keep shipping clean code`
                  : "Run your first analysis to see ML insights across 12 models"}
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 shrink-0">
              {!isReviewer && recentEntries.length > 0 && (
                <Button variant="outline" className="gap-2 shadow-sm" onClick={() => navigate("/compare")}>
                  <GitCompare className="w-4 h-4" /> Compare
                </Button>
              )}
              {isReviewer ? (
                <Button className="gap-2 shadow-lg" onClick={() => navigate("/reviews")}>
                  <ClipboardCheck className="w-4 h-4" /> Review Queue
                </Button>
              ) : (
                <Button className="gap-2 shadow-lg" onClick={() => navigate("/submit")}>
                  <Plus className="w-4 h-4" /> New Analysis
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* ── Reviewer Queue ── */}
        {isReviewer && pendingReviewEntries.length > 0 && (
          <div className="bg-card border border-yellow-500/25 rounded-xl mb-8 overflow-hidden">
            <div className="flex items-center justify-between px-6 py-3.5 border-b border-border bg-yellow-500/5">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500" />
                <h2 className="font-semibold text-foreground text-sm">Pending Review Queue</h2>
                <span className="bg-yellow-500/20 text-yellow-400 text-xs font-bold px-2 py-0.5 rounded-full">
                  {pendingReviewEntries.length}
                </span>
              </div>
              <Button variant="outline" size="sm" onClick={() => navigate("/reviews")}>View All</Button>
            </div>
            <div className="divide-y divide-border">
              {pendingReviewEntries.slice(0, 3).map((entry) => {
                const fbKeys = loadFeedbackKeys();
                const unreviewedCount = (entry.result?.security?.findings ?? []).filter(
                  (f: { title: string; lineno: number }) => !fbKeys.has(`${f.title}-${f.lineno}`)
                ).length;
                return (
                  <div key={entry.id} className="flex items-center gap-4 px-6 py-3.5 hover:bg-secondary/10 transition-colors">
                    <div className="w-8 h-8 rounded-lg bg-yellow-500/10 flex items-center justify-center shrink-0">
                      <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">{entry.filename}</p>
                      <p className="text-xs text-muted-foreground">{fmt(entry.submittedAt)} · {unreviewedCount} finding{unreviewedCount !== 1 ? "s" : ""} pending</p>
                    </div>
                    <Button size="sm" className="gap-1 text-xs shrink-0" onClick={() => navigate(`/reviews/${entry.id}`)}>
                      <ClipboardCheck className="w-3.5 h-3.5" /> Review
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Stat Cards ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {[
            {
              Icon: BarChart3, iconCls: "text-primary",   iconBg: "bg-primary/10",    border: "border-l-primary",
              label: "Total Reviews", value: realStats ? String(realStats.totalReviews) : "0",
              sub: realStats ? `+${realStats.thisWeek} this week` : "No analyses yet",
            },
            {
              Icon: Shield, iconCls: "text-orange-400", iconBg: "bg-orange-500/10", border: "border-l-orange-500",
              label: "Issues Found", value: realStats ? realStats.issuesFound.toLocaleString() : "0",
              sub: realStats ? "across all analyses" : "Submit code to start",
            },
            {
              Icon: Zap, iconCls: "text-yellow-400", iconBg: "bg-yellow-500/10", border: "border-l-yellow-500",
              label: "Avg Review Time", value: "< 30s",
              sub: "AI-powered instant",
            },
            {
              Icon: TrendingUp, iconCls: "text-emerald-400", iconBg: "bg-emerald-500/10", border: "border-l-emerald-500",
              label: "Quality Score", value: realStats ? `${realStats.avgScore}` : "—",
              sub: realStats ? "avg across analyses" : "Run an analysis to see",
              suffix: realStats ? "/100" : "",
            },
          ].map((c) => (
            <div key={c.label} className={`bg-card border border-border border-l-4 ${c.border} rounded-xl p-5 flex items-center gap-4 hover:shadow-md transition-all hover:-translate-y-0.5`}>
              <div className={`p-2.5 ${c.iconBg} rounded-xl shrink-0`}>
                <c.Icon className={`w-5 h-5 ${c.iconCls}`} />
              </div>
              <div className="min-w-0">
                <p className="text-2xl font-bold text-foreground leading-none mb-0.5">
                  {c.value}
                  {c.suffix && <span className="text-sm text-muted-foreground font-normal ml-0.5">{c.suffix}</span>}
                </p>
                <p className="text-xs font-medium text-muted-foreground">{c.label}</p>
                <p className="text-[11px] text-muted-foreground/60 mt-0.5 truncate">{c.sub}</p>
              </div>
            </div>
          ))}
        </div>

        {/* ── Quality Trend ── */}
        {trendData.length >= 2 && (
          <div className="bg-card border border-border rounded-xl mb-8 p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                <h2 className="font-semibold text-foreground text-sm">Quality Score Trend</h2>
                <span className="text-xs text-muted-foreground">last {trendData.length} analyses</span>
              </div>
              <Button variant="outline" size="sm" onClick={() => navigate("/analytics")}>
                Full Analytics
              </Button>
            </div>
            <div className="h-[140px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="date" fontSize={10} stroke="hsl(var(--muted-foreground))" tick={{ fill: "hsl(var(--muted-foreground))" }} />
                  <YAxis domain={[0, 100]} fontSize={10} stroke="hsl(var(--muted-foreground))" tick={{ fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "12px" }}
                    labelStyle={{ color: "hsl(var(--foreground))" }}
                  />
                  <Area
                    type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={2}
                    fill="url(#scoreGrad)"
                    dot={{ fill: "hsl(var(--primary))", r: 3, strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: "hsl(var(--primary))" }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── Leaderboard ── */}
        {latestByFile.length >= 2 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Best files */}
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="flex items-center gap-2 px-5 py-3.5 border-b border-border bg-emerald-500/5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <h2 className="font-semibold text-foreground text-sm">Top Quality Files</h2>
              </div>
              <div className="divide-y divide-border">
                {topFiles.map((entry, i) => (
                  <div key={entry.id} className="flex items-center gap-3 px-5 py-3 hover:bg-secondary/10 cursor-pointer transition-colors" onClick={() => navigate(`/reviews/${entry.id}`)}>
                    <span className="text-xs font-bold text-muted-foreground w-4 shrink-0">{i + 1}</span>
                    <span className="flex-1 text-sm text-foreground font-mono truncate">{entry.filename}</span>
                    <ScoreMini score={entry.overallScore} />
                  </div>
                ))}
              </div>
            </div>
            {/* Worst files (needs attention) */}
            {worstFiles.length > 0 && (
              <div className="bg-card border border-border rounded-xl overflow-hidden">
                <div className="flex items-center gap-2 px-5 py-3.5 border-b border-border bg-orange-500/5">
                  <AlertTriangle className="w-4 h-4 text-orange-400" />
                  <h2 className="font-semibold text-foreground text-sm">Needs Attention</h2>
                </div>
                <div className="divide-y divide-border">
                  {worstFiles.map((entry, i) => (
                    <div key={entry.id} className="flex items-center gap-3 px-5 py-3 hover:bg-secondary/10 cursor-pointer transition-colors" onClick={() => navigate(`/reviews/${entry.id}`)}>
                      <span className="text-xs font-bold text-muted-foreground w-4 shrink-0">{i + 1}</span>
                      <span className="flex-1 text-sm text-foreground font-mono truncate">{entry.filename}</span>
                      <ScoreMini score={entry.overallScore} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Recent Reviews ── */}
        <div className="bg-card border border-border rounded-xl mb-8 overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-border">
            <div className="flex items-center gap-2">
              <FileCode2 className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">Recent Reviews</h2>
              {recentEntries.length > 0 && (
                <span className="bg-secondary text-muted-foreground text-xs px-2 py-0.5 rounded-full font-medium">
                  {recentEntries.length}
                </span>
              )}
            </div>
            <Button variant="outline" size="sm" onClick={() => navigate("/reviews")}>View All</Button>
          </div>

          {recentEntries.length === 0 ? (
            <div className="px-6 py-16 text-center">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <FileCode2 className="w-8 h-8 text-primary/40" />
              </div>
              <p className="text-sm font-medium text-foreground mb-1">No analyses yet</p>
              <p className="text-xs text-muted-foreground mb-5">Submit a code snippet and all 12 ML models will analyze it instantly.</p>
              <Button size="sm" className="gap-2" onClick={() => navigate("/submit")}>
                <Plus className="w-3.5 h-3.5" /> New Analysis
              </Button>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {recentEntries.map((entry) => {
                const cfg = SEV_CONFIG[entry.severity] ?? SEV_CONFIG.none;
                return (
                  <div
                    key={entry.id}
                    className="flex items-center gap-4 px-6 py-4 hover:bg-secondary/10 transition-colors group cursor-pointer"
                    onClick={() => navigate(`/reviews/${entry.id}`)}
                  >
                    {/* severity dot */}
                    <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${cfg.bar}`} />

                    {/* info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap mb-0.5">
                        <span className="text-sm font-medium text-foreground truncate">{entry.filename}</span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${cfg.cls}`}>
                          {entry.severity === "none" ? "Clean" : entry.severity}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <span>{fmt(entry.submittedAt)}</span>
                        <span>·</span>
                        <span>{entry.issueCount} issues</span>
                        <span>·</span>
                        <span className={`capitalize font-medium ${STATUS_CLS[entry.status]}`}>
                          {entry.status.replace("_", " ")}
                        </span>
                      </div>
                    </div>

                    {/* score */}
                    <ScoreMini score={entry.overallScore} />

                    {/* action button */}
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1 text-xs opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                      onClick={(e) => { e.stopPropagation(); navigate(`/reviews/${entry.id}`); }}
                    >
                      <ExternalLink className="w-3.5 h-3.5" /> View
                    </Button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <QuickActions />
      </main>
    </div>
  );
};

export default Dashboard;
