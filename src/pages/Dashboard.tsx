import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { QuickActions } from "@/components/app/QuickActions";
import { Button } from "@/components/ui/button";
import { getSession } from "@/services/auth";
import { getEntries, getDashboardStats } from "@/services/reviewHistory";
import { STORAGE_KEYS } from "@/constants/storage";
import {
  ExternalLink, Clock, FileCode2, AlertTriangle, ClipboardCheck,
  CheckCircle2, TrendingUp, Shield, Zap, BarChart3, Plus, GitCompare,
  Flame, X, Activity,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

function savedFirstName(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.settings);
    const name = raw ? JSON.parse(raw)?.name : null;
    if (name) return name.split(" ")[0];
  } catch { /* ignore */ }
  // Fall back to session name, then a generic greeting
  const session = getSession();
  return session?.name?.split(" ")[0] ?? "there";
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
    const raw = localStorage.getItem(STORAGE_KEYS.feedback);
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

// ─── Activity Heatmap ─────────────────────────────────────────────────────────

function ActivityHeatmap({ entries }: { entries: ReturnType<typeof getEntries> }) {
  const WEEKS = 15;
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  // Anchor to the start of the current week (Monday)
  const dayOfWeek = (today.getDay() + 6) % 7; // Mon=0 … Sun=6
  const weekStart = new Date(today);
  weekStart.setDate(today.getDate() - dayOfWeek);

  // Grid start = (WEEKS-1) weeks before weekStart
  const gridStart = new Date(weekStart);
  gridStart.setDate(weekStart.getDate() - (WEEKS - 1) * 7);

  // Build date → count map
  const countMap: Record<string, number> = {};
  entries.forEach((e) => {
    const d = new Date(e.submittedAt);
    d.setHours(0, 0, 0, 0);
    const key = d.toISOString().slice(0, 10);
    countMap[key] = (countMap[key] ?? 0) + 1;
  });

  // Build cells: WEEKS columns, 7 rows (Mon…Sun)
  const cells: { date: Date; count: number }[][] = Array.from({ length: WEEKS }, (_, wi) =>
    Array.from({ length: 7 }, (_, di) => {
      const d = new Date(gridStart);
      d.setDate(gridStart.getDate() + wi * 7 + di);
      const key = d.toISOString().slice(0, 10);
      return { date: d, count: countMap[key] ?? 0 };
    })
  );

  const maxCount = Math.max(1, ...Object.values(countMap));

  function cellColor(count: number) {
    if (count === 0) return "bg-secondary/40";
    const intensity = Math.ceil((count / maxCount) * 4);
    return [
      "bg-primary/20",
      "bg-primary/40",
      "bg-primary/65",
      "bg-primary",
    ][Math.min(intensity - 1, 3)];
  }

  const totalThisPeriod = cells.flat().reduce((s, c) => s + c.count, 0);
  const DAY_LABELS = ["Mon", "", "Wed", "", "Fri", "", "Sun"];

  return (
    <div className="bg-card border border-border rounded-xl p-5 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <h2 className="font-semibold text-foreground text-sm">Analysis Activity</h2>
          <span className="text-xs text-muted-foreground">last {WEEKS} weeks</span>
        </div>
        <span className="text-xs text-muted-foreground">{totalThisPeriod} analyses</span>
      </div>

      <div className="flex gap-1">
        {/* Day labels */}
        <div className="flex flex-col gap-0.5 justify-between mr-1" style={{ paddingTop: "0px" }}>
          {DAY_LABELS.map((l, i) => (
            <div key={i} className="text-[9px] text-muted-foreground/50 leading-none" style={{ height: "11px", lineHeight: "11px" }}>{l}</div>
          ))}
        </div>

        {/* Grid */}
        {cells.map((week, wi) => (
          <div key={wi} className="flex flex-col gap-0.5">
            {week.map((cell, di) => {
              const isFuture = cell.date > today;
              const label = cell.date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
              return (
                <div
                  key={di}
                  title={isFuture ? "" : `${label}: ${cell.count} analysis${cell.count !== 1 ? "es" : ""}`}
                  className={`w-2.5 h-2.5 rounded-sm transition-colors ${isFuture ? "bg-transparent" : cellColor(cell.count)}`}
                />
              );
            })}
          </div>
        ))}

        {/* Legend */}
        <div className="flex flex-col justify-end ml-2 gap-0.5">
          <span className="text-[9px] text-muted-foreground/50 mb-0.5">Less</span>
          {["bg-secondary/40", "bg-primary/20", "bg-primary/40", "bg-primary/65", "bg-primary"].map((cls, i) => (
            <div key={i} className={`w-2.5 h-2.5 rounded-sm ${cls}`} />
          ))}
          <span className="text-[9px] text-muted-foreground/50 mt-0.5">More</span>
        </div>
      </div>
    </div>
  );
}

// ─── Grade Ring ───────────────────────────────────────────────────────────────

function GradeRing({ score }: { score: number }) {
  const grade = score >= 90 ? "A" : score >= 80 ? "B" : score >= 70 ? "C" : score >= 60 ? "D" : "F";
  const color = score >= 90 ? "#10b981" : score >= 80 ? "#22c55e" : score >= 70 ? "#eab308" : score >= 60 ? "#f97316" : "#ef4444";
  const SIZE = 88, R = 34, CIRC = 2 * Math.PI * R;
  return (
    <div className="relative hidden sm:flex items-center justify-center shrink-0">
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        <circle cx={SIZE / 2} cy={SIZE / 2} r={R} fill="none" stroke="hsl(var(--border))" strokeWidth="6" />
        <circle
          cx={SIZE / 2} cy={SIZE / 2} r={R} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={CIRC}
          strokeDashoffset={CIRC * (1 - score / 100)}
          strokeLinecap="round"
          transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
        />
      </svg>
      <div className="absolute flex flex-col items-center leading-none gap-0.5">
        <span className="text-2xl font-black leading-none" style={{ color }}>{grade}</span>
        <span className="text-[10px] text-muted-foreground leading-none">{score}/100</span>
      </div>
    </div>
  );
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
    const onVisible = () => {
      if (document.visibilityState === "visible") setHistoryVersion((v) => v + 1);
    };
    window.addEventListener("storage", onStorage);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("storage", onStorage);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const realStats = useMemo(() => getDashboardStats(), [historyVersion]);
  const allEntries = useMemo(() => getEntries(), [historyVersion]);
  const recentEntries = allEntries.slice(0, 5);
  // Dedup by repo+filename — keep the latest score per unique file
  const latestByFile = useMemo(() => {
    const seen = new Map<string, typeof allEntries[number]>();
    allEntries.forEach((e) => {
      const key = `${e.repoName ?? ""}::${e.filename}`;
      if (!seen.has(key)) seen.set(key, e);
    });
    return [...seen.values()];
  }, [allEntries]);
  const topFiles  = [...latestByFile].sort((a, b) => b.overallScore - a.overallScore).slice(0, 5);
  const worstFiles = [...latestByFile].sort((a, b) => a.overallScore - b.overallScore).filter(f => f.overallScore < 80).slice(0, 5);
  const trendData = useMemo(
    () =>
      [...allEntries]
        .reverse()
        .slice(-20)
        .map((e) => ({
          date: new Date(e.submittedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
          score: e.overallScore,
        })),
    [allEntries]
  );

  const [urgentDismissed, setUrgentDismissed] = useState(
    () => localStorage.getItem("intellcode_urgent_dismissed") === "1"
  );
  const urgentItems = useMemo(() => {
    const items: { label: string; entryId: string; detail: string }[] = [];
    allEntries.forEach((e) => {
      if (e.severity === "critical")
        items.push({ label: e.filename, entryId: e.id, detail: `Critical severity · score ${e.overallScore}` });
      else if (e.severity === "high" && e.overallScore < 60)
        items.push({ label: e.filename, entryId: e.id, detail: `High severity · score ${e.overallScore}` });
    });
    return items;
  }, [allEntries]);

  const reviewerFbKeys = useMemo(
    () => (isReviewer ? loadFeedbackKeys() : new Set<string>()),
    [historyVersion, isReviewer]
  );
  const pendingReviewEntries = useMemo(() => {
    if (!isReviewer) return [];
    return allEntries.filter((e) => {
      const findings = e.result?.security?.findings ?? [];
      if (findings.length === 0) return false;
      return findings.some(
        (f: { title: string; lineno: number }) => !reviewerFbKeys.has(`${f.title}-${f.lineno}`)
      );
    });
  }, [allEntries, isReviewer, reviewerFbKeys]);

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
                  : "Run your first analysis to see ML insights"}
              </p>
            </div>
            {realStats && <GradeRing score={realStats.avgScore} />}
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
              <Button variant="outline" size="sm" onClick={() => navigate("/reviews")}>
                View All{pendingReviewEntries.length > 3 ? ` (${pendingReviewEntries.length - 3} more)` : ""}
              </Button>
            </div>
            <div className="divide-y divide-border">
              {pendingReviewEntries.slice(0, 3).map((entry) => {
                const unreviewedCount = (entry.result?.security?.findings ?? []).filter(
                  (f: { title: string; lineno: number }) => !reviewerFbKeys.has(`${f.title}-${f.lineno}`)
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

        {/* ── Urgent Actions ── */}
        {!urgentDismissed && urgentItems.length > 0 && (
          <div className="bg-red-500/5 border border-red-500/20 rounded-xl mb-8 overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-red-500/20">
              <div className="flex items-center gap-2">
                <Flame className="w-4 h-4 text-red-400" />
                <h2 className="font-semibold text-foreground text-sm">Urgent Actions</h2>
                <span className="bg-red-500/20 text-red-400 text-xs font-bold px-2 py-0.5 rounded-full">{urgentItems.length}</span>
                {urgentItems.length > 5 && <span className="text-xs text-muted-foreground">showing top 5</span>}
              </div>
              <button onClick={() => { setUrgentDismissed(true); localStorage.setItem("intellcode_urgent_dismissed", "1"); }} className="text-muted-foreground hover:text-foreground">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="divide-y divide-red-500/10">
              {urgentItems.slice(0, 5).map((item) => (
                <div
                  key={item.entryId}
                  className="flex items-center gap-3 px-5 py-3 hover:bg-red-500/5 cursor-pointer transition-colors"
                  onClick={() => navigate(`/reviews/${item.entryId}`)}
                >
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{item.label}</p>
                    <p className="text-xs text-muted-foreground">{item.detail}</p>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Quality Trend ── */}
        {trendData.length >= 2 && (
          <div className="bg-card border border-border rounded-xl mb-8 p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                <h2 className="font-semibold text-foreground text-sm">Quality Score Trend</h2>
                <span className="text-xs text-muted-foreground">
                  {allEntries.length > 20 ? `last 20 of ${allEntries.length} analyses` : `last ${trendData.length} analyses`}
                </span>
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

        {/* ── Activity Heatmap ── */}
        {allEntries.length >= 2 && <ActivityHeatmap entries={allEntries} />}

        {/* ── Leaderboard ── */}
        {latestByFile.length >= 2 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Best files */}
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-border bg-emerald-500/5">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <h2 className="font-semibold text-foreground text-sm">Top Quality Files</h2>
                  {latestByFile.length > 5 && <span className="text-xs text-muted-foreground">top 5 of {latestByFile.length}</span>}
                </div>
                {latestByFile.length > 5 && (
                  <Button variant="ghost" size="sm" className="text-xs h-7 px-2" onClick={() => navigate("/reviews")}>View All</Button>
                )}
              </div>
              <div className="divide-y divide-border">
                {topFiles.map((entry, i) => (
                  <div key={entry.id} className="flex items-center gap-3 px-5 py-3 hover:bg-secondary/10 cursor-pointer transition-colors" onClick={() => navigate(`/reviews/${entry.id}`)}>
                    <span className="text-xs font-bold text-muted-foreground w-4 shrink-0">{i + 1}</span>
                    <div className="flex-1 min-w-0">
                      <span className="text-sm text-foreground font-mono truncate block">{entry.filename}</span>
                      {entry.repoName && <span className="text-[10px] text-muted-foreground truncate block">{entry.repoName}</span>}
                    </div>
                    <ScoreMini score={entry.overallScore} />
                  </div>
                ))}
              </div>
            </div>
            {/* Worst files (needs attention) */}
            {worstFiles.length > 0 && (
              <div className="bg-card border border-border rounded-xl overflow-hidden">
                <div className="flex items-center justify-between px-5 py-3.5 border-b border-border bg-orange-500/5">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-orange-400" />
                    <h2 className="font-semibold text-foreground text-sm">Needs Attention</h2>
                    {latestByFile.filter(f => f.overallScore < 80).length > 5 && (
                      <span className="text-xs text-muted-foreground">top 5 of {latestByFile.filter(f => f.overallScore < 80).length}</span>
                    )}
                  </div>
                  {latestByFile.filter(f => f.overallScore < 80).length > 5 && (
                    <Button variant="ghost" size="sm" className="text-xs h-7 px-2" onClick={() => navigate("/reviews")}>View All</Button>
                  )}
                </div>
                <div className="divide-y divide-border">
                  {worstFiles.map((entry, i) => (
                    <div key={entry.id} className="flex items-center gap-3 px-5 py-3 hover:bg-secondary/10 cursor-pointer transition-colors" onClick={() => navigate(`/reviews/${entry.id}`)}>
                      <span className="text-xs font-bold text-muted-foreground w-4 shrink-0">{i + 1}</span>
                      <div className="flex-1 min-w-0">
                        <span className="text-sm text-foreground font-mono truncate block">{entry.filename}</span>
                        {entry.repoName && <span className="text-[10px] text-muted-foreground truncate block">{entry.repoName}</span>}
                      </div>
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
              <p className="text-xs text-muted-foreground mb-5">Submit a code snippet and our ML models will analyze it instantly.</p>
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
