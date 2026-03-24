import { useState, useMemo, useEffect } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { StatCard } from "@/components/app/StatCard";
import { Button } from "@/components/ui/button";
import { Download, Lightbulb, TrendingUp, BarChart3, PieChart as PieIcon } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import {
  mockAnalyticsStats,
  mockQualityTrend,
  mockIssueDistribution,
  mockCommonIssues,
  mockKeyInsights,
} from "@/data/mockData";
import { getEntries } from "@/services/reviewHistory";
import { toast } from "sonner";

// ─── Derive real analytics from localStorage history ─────────────────────────

function buildAnalytics(range: number) {
  const all = getEntries();
  if (all.length === 0) return null;

  const cutoff = Date.now() - range * 24 * 60 * 60 * 1000;
  const entries = all.filter((e) => new Date(e.submittedAt).getTime() >= cutoff);
  const usedFallback = entries.length === 0;
  const use = usedFallback ? all : entries;

  // Quality trend (group by day)
  const byDay: Record<string, number[]> = {};
  use.forEach((e) => {
    const day = new Date(e.submittedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" });
    byDay[day] = byDay[day] ?? [];
    byDay[day].push(e.overallScore);
  });
  const qualityTrend = Object.entries(byDay)
    .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
    .map(([date, scores]) => ({
      date,
      score: Math.round(scores.reduce((s, v) => s + v, 0) / scores.length),
    }));

  // Issue distribution by severity
  const sevCounts: Record<string, number> = { critical: 0, high: 0, medium: 0, low: 0 };
  use.forEach((e) => { if (e.severity !== "none") sevCounts[e.severity] = (sevCounts[e.severity] ?? 0) + 1; });
  const PIE_COLORS: Record<string, string> = {
    critical: "hsl(0, 85%, 60%)",
    high:     "hsl(25, 95%, 53%)",
    medium:   "hsl(38, 92%, 50%)",
    low:      "hsl(210, 90%, 60%)",
  };
  const issueDistribution = Object.entries(sevCounts)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name: name[0].toUpperCase() + name.slice(1), value, color: PIE_COLORS[name] }));

  // Common issues by category (from all issues across entries)
  const catCounts: Record<string, number> = {};
  use.forEach((e) => {
    const r = e.result;
    const add = (cat: string, n: number) => { catCounts[cat] = (catCounts[cat] ?? 0) + n; };
    add("Security",     r.security?.findings?.length ?? 0);
    add("Dead Code",    r.dead_code?.issues?.length ?? 0);
    add("Performance",  r.performance?.issues?.length ?? 0);
    add("Refactoring",  r.refactoring?.suggestions?.length ?? 0);
    add("Dependencies", r.dependencies?.issues?.length ?? 0);
    add("Clones",       r.clones?.clones?.length ?? 0);
  });
  const BAR_COLORS = ["hsl(210,90%,60%)", "hsl(142,76%,36%)", "hsl(38,92%,50%)", "hsl(25,95%,53%)", "hsl(0,85%,60%)", "hsl(270,80%,60%)", "hsl(180,70%,45%)"];
  const commonIssues = Object.entries(catCounts)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], i) => ({ name, count, color: BAR_COLORS[i % BAR_COLORS.length] }));

  const avgScore = Math.round(use.reduce((s, e) => s + e.overallScore, 0) / use.length);
  const totalIssues = use.reduce((s, e) => s + e.issueCount, 0);

  // Score distribution histogram (buckets: 0-20, 20-40, 40-60, 60-80, 80-100)
  const scoreBuckets = [
    { range: "0–20",  count: 0, color: "hsl(0,85%,60%)" },
    { range: "20–40", count: 0, color: "hsl(25,95%,53%)" },
    { range: "40–60", count: 0, color: "hsl(38,92%,50%)" },
    { range: "60–80", count: 0, color: "hsl(210,90%,60%)" },
    { range: "80–100",count: 0, color: "hsl(142,76%,36%)" },
  ];
  use.forEach((e) => {
    const idx = Math.min(4, Math.floor(e.overallScore / 20));
    scoreBuckets[idx].count++;
  });
  const insights: string[] = [];
  if (avgScore >= 80) insights.push(`Average code quality is strong at ${avgScore}/100`);
  else insights.push(`Average code quality is ${avgScore}/100 — room for improvement`);
  if (sevCounts.critical > 0) insights.push(`${sevCounts.critical} critical severity file${sevCounts.critical > 1 ? "s" : ""} detected — prioritize fixes`);
  if (catCounts["Security"] > 0) insights.push(`${catCounts["Security"]} security issue${catCounts["Security"] > 1 ? "s" : ""} found — review OWASP risks`);
  insights.push(`${use.length} analysis${use.length > 1 ? "es" : ""} run in selected period`);

  // Per-file score history (files analyzed more than once)
  const byFile: Record<string, { score: number; date: string }[]> = {};
  [...use].reverse().forEach((e) => {
    const key = e.filename;
    byFile[key] = byFile[key] ?? [];
    byFile[key].push({
      score: e.overallScore,
      date: new Date(e.submittedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    });
  });
  const fileTimelines = Object.entries(byFile)
    .filter(([, runs]) => runs.length >= 2)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 5)
    .map(([filename, runs]) => ({
      filename,
      runs,
      delta: runs[runs.length - 1].score - runs[0].score,
      latest: runs[runs.length - 1].score,
    }));

  return { qualityTrend, issueDistribution, commonIssues, scoreBuckets, avgScore, totalIssues, insights, fileTimelines, count: use.length, usedFallback };
}

// ─── Component ───────────────────────────────────────────────────────────────

const Analytics = () => {
  const [timeRange, setTimeRange] = useState<7 | 30 | 90>(30);
  const [historyVersion, setHistoryVersion] = useState(0);

  // Recompute when timeRange changes OR when new analyses are added (storage changes)
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "intellcode_review_history") setHistoryVersion((v) => v + 1);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const real = useMemo(() => buildAnalytics(timeRange), [timeRange, historyVersion]);

  const qualityTrend   = real?.qualityTrend   ?? mockQualityTrend;
  const issueDist      = real ? real.issueDistribution : mockIssueDistribution;
  const commonIssues   = real?.commonIssues.length ? real.commonIssues : mockCommonIssues;
  const scoreBuckets   = real?.scoreBuckets ?? null;
  const fileTimelines  = real?.fileTimelines ?? [];
  const insights       = real?.insights       ?? mockKeyInsights;

  const totalReviews   = real?.count          ?? mockAnalyticsStats.totalReviews;
  const avgScore       = real?.avgScore       ?? mockAnalyticsStats.avgQualityScore;
  const totalIssues    = real?.totalIssues    ?? 312;
  const usingReal      = !!real;
  const usedFallback   = real?.usedFallback ?? false;

  const handleExport = () => {
    const entries = getEntries();
    const blob = new Blob([JSON.stringify({ generated: new Date().toISOString(), range: `${timeRange}d`, entries }, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `intellcode_analytics_${Date.now()}.json` });
    a.click();
    URL.revokeObjectURL(a.href);
    toast.success("Analytics exported as JSON");
  };

  const TOOLTIP_STYLE = {
    contentStyle: { backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px" },
    labelStyle: { color: "hsl(var(--foreground))" },
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Analytics & Insights</h1>
            {!usingReal && (
              <p className="text-xs text-muted-foreground mt-1">
                Showing sample data — run an analysis to see real metrics
              </p>
            )}
            {usingReal && usedFallback && (
              <p className="text-xs text-yellow-500 mt-1">
                No data in last {timeRange} days — showing all-time results instead
              </p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {([7, 30, 90] as const).map((r) => (
              <Button key={r} variant={timeRange === r ? "default" : "outline"} size="sm" onClick={() => setTimeRange(r)}>
                {r} Days
              </Button>
            ))}
            <Button size="sm" className="bg-gradient-primary gap-1.5" onClick={handleExport}>
              <Download className="w-4 h-4" /> Export
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <StatCard title="Total Reviews" value={totalReviews} change={usingReal ? (usedFallback ? "all-time" : `last ${timeRange} days`) : "+23% vs prev period"} changeType="positive" />
          <StatCard title="Avg Quality Score" value={avgScore} suffix="/100" change={usingReal ? "across all analyses" : "+4.2 vs baseline"} changeType="positive" />
          <StatCard title="Total Issues Found" value={totalIssues} change={usingReal ? "across all analyses" : "sample data"} changeType="neutral" />
        </div>

        {/* Quality Trend */}
        <div className="bg-card border border-border rounded-xl p-6 mb-8">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-primary" />
            <h2 className="text-lg font-semibold text-primary">Code Quality Trend</h2>
          </div>
          <p className="text-sm text-muted-foreground mb-6">Average quality score over time</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={qualityTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} domain={[0, 100]} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Line type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={3}
                  dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, fill: "hsl(var(--primary))" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Pie — severity distribution */}
          <div className="bg-card border border-border rounded-xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <PieIcon className="w-4 h-4 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">Issue Distribution</h2>
            </div>
            <div className="flex items-center gap-4">
              <div className="h-[220px] flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={issueDist} cx="50%" cy="50%" innerRadius={55} outerRadius={90}
                      dataKey="value" paddingAngle={3}>
                      {issueDist.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                    </Pie>
                    <Tooltip {...TOOLTIP_STYLE} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2">
                {issueDist.map((item) => (
                  <div key={item.name} className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                    <span className="text-muted-foreground">{item.name}</span>
                    <span className="font-semibold text-foreground ml-auto">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Bar — common issues by category */}
          <div className="bg-card border border-border rounded-xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">Issues by Category</h2>
            </div>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={commonIssues} layout="vertical" margin={{ left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} width={90} />
                  <Tooltip {...TOOLTIP_STYLE} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {commonIssues.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Score Distribution Histogram */}
        {scoreBuckets && scoreBuckets.some((b) => b.count > 0) && (
          <div className="bg-card border border-border rounded-xl p-6 mb-8">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">Score Distribution</h2>
              <span className="text-xs text-muted-foreground ml-1">files by quality score bucket</span>
            </div>
            <div className="h-[180px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scoreBuckets} margin={{ left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="range" stroke="hsl(var(--muted-foreground))" fontSize={11} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} allowDecimals={false} />
                  <Tooltip
                    {...TOOLTIP_STYLE}
                    formatter={(value: number) => [`${value} file${value !== 1 ? "s" : ""}`, "Count"]}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {scoreBuckets.map((b) => (
                      <Cell key={b.range} fill={b.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Per-file timelines — only shown when files have 2+ analyses */}
        {fileTimelines.length > 0 && (
          <div className="bg-card border border-border rounded-xl p-6 mb-8">
            <div className="flex items-center gap-2 mb-5">
              <TrendingUp className="w-4 h-4 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">File Quality Progress</h2>
              <span className="text-xs text-muted-foreground ml-1">files analyzed 2+ times</span>
            </div>
            <div className="space-y-5">
              {fileTimelines.map(({ filename, runs, delta, latest }) => {
                const improved = delta > 0;
                const unchanged = delta === 0;
                return (
                  <div key={filename}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-mono text-foreground truncate max-w-[60%]">{filename}</span>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-semibold ${improved ? "text-emerald-400" : unchanged ? "text-muted-foreground" : "text-red-400"}`}>
                          {improved ? `+${delta}` : delta} pts
                        </span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${latest >= 80 ? "bg-emerald-500/20 text-emerald-400" : latest >= 60 ? "bg-yellow-500/20 text-yellow-400" : "bg-red-500/20 text-red-400"}`}>
                          {latest}/100
                        </span>
                      </div>
                    </div>
                    <div className="h-[60px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={runs} margin={{ top: 4, right: 4, bottom: 0, left: -30 }}>
                          <XAxis dataKey="date" fontSize={9} stroke="hsl(var(--muted-foreground))" />
                          <YAxis domain={[0, 100]} fontSize={9} stroke="hsl(var(--muted-foreground))" />
                          <Tooltip {...TOOLTIP_STYLE} />
                          <Line type="monotone" dataKey="score" stroke={improved ? "hsl(142,76%,36%)" : unchanged ? "hsl(var(--primary))" : "hsl(0,85%,60%)"}
                            strokeWidth={2} dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Key Insights */}
        <div className="bg-card border border-border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            <h2 className="text-lg font-semibold text-foreground">Key Insights</h2>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            {insights.map((insight, i) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-secondary/30 border border-border rounded-lg">
                <div className="w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </div>
                <p className="text-sm text-foreground">{insight}</p>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Analytics;
