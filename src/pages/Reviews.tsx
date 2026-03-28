import { useState, useEffect, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Download,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  CheckCircle2,
  XCircle,
  ExternalLink,
  AlertTriangle,
  ShieldAlert,
  Bug,
  Trash2,
  Clock,
  FileCode2,
  FolderOpen,
  Folder,
  ArrowLeft,
  Github,
} from "lucide-react";
import { toast } from "sonner";
import {
  getEntries,
  markResolved,
  markFalsePositive,
  clearHistory,
  deleteEntry,
  type HistoryEntry,
} from "@/services/reviewHistory";


// ─── Helpers ──────────────────────────────────────────────────────────────────

const SEV_CLS: Record<string, string> = {
  critical: "bg-red-600 text-white",
  high:     "bg-orange-500 text-white",
  medium:   "bg-yellow-500 text-black",
  low:      "bg-blue-500 text-white",
  none:     "bg-muted text-muted-foreground",
};

const SEV_BORDER: Record<string, string> = {
  critical: "border-l-red-600",
  high:     "border-l-orange-500",
  medium:   "border-l-yellow-500",
  low:      "border-l-blue-500",
  none:     "border-l-emerald-500",
};

const SEV_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, none: 4 };

function ScoreMini({ score }: { score: number }) {
  const color = score >= 80 ? "bg-emerald-500" : score >= 60 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 shrink-0">
      <div className="w-16 h-1.5 bg-secondary rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs font-semibold tabular-nums text-foreground w-8 text-right">{score}/100</span>
    </div>
  );
}

const STATUS_CLS: Record<string, string> = {
  completed:   "bg-green-500/20 text-green-400 border-green-500/30",
  in_progress: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  pending:     "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function flatIssues(entry: HistoryEntry) {
  const r = entry.result;
  const items: { key: string; severity: string; category: string; title: string; description: string }[] = [];
  (r.security?.findings ?? []).forEach((f, i) =>
    items.push({ key: `security_${i}`, severity: f.severity ?? "medium", category: "Security", title: f.title ?? f.vuln_type ?? "Security Issue", description: f.description ?? "" })
  );
  if (r.bug_prediction?.risk_level && r.bug_prediction.risk_level !== "low") {
    items.push({ key: "bug_risk", severity: r.bug_prediction.risk_level === "critical" ? "critical" : r.bug_prediction.risk_level === "high" ? "high" : "medium", category: "Bug Risk", title: `Bug probability: ${Math.round((r.bug_prediction.bug_probability ?? 0) * 100)}% (${r.bug_prediction.risk_level})`, description: (r.bug_prediction.risk_factors ?? []).join("; ") });
  }
  (r.dead_code?.issues ?? []).forEach((iss, i) =>
    items.push({ key: `dead_${i}`, severity: "low", category: "Dead Code", title: iss.title ?? iss.issue_type ?? "Dead Code", description: iss.description ?? "" })
  );
  (r.refactoring?.suggestions ?? []).forEach((s, i) =>
    items.push({ key: `refactor_${i}`, severity: s.priority ?? "medium", category: "Refactoring", title: s.title ?? s.refactoring_type ?? "Refactoring", description: s.description ?? "" })
  );
  (r.performance?.issues ?? []).forEach((h, i) =>
    items.push({ key: `perf_${i}`, severity: h.severity ?? "medium", category: "Performance", title: h.title ?? h.pattern_type ?? "Performance", description: h.description ?? "" })
  );
  (r.dependencies?.issues ?? []).forEach((d, i) =>
    items.push({ key: `dep_${i}`, severity: d.severity ?? "low", category: "Dependency", title: d.title ?? d.issue_type ?? "Dependency Issue", description: d.description ?? "" })
  );
  return items;
}

// ─── Issue Row ────────────────────────────────────────────────────────────────

function IssueRow({
  entryId, issue, resolved, fp, onAction,
}: {
  entryId: string;
  issue: ReturnType<typeof flatIssues>[number];
  resolved: boolean;
  fp: boolean;
  onAction: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`border rounded-lg mb-2 transition-all ${resolved ? "border-green-500/30 bg-green-500/5 opacity-60" : fp ? "border-muted bg-muted/20 opacity-50" : "border-border bg-card"}`}>
      <div className="flex items-center gap-3 p-3 cursor-pointer" onClick={() => setOpen((o) => !o)}>
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${SEV_CLS[issue.severity] ?? SEV_CLS.medium}`}>
          {issue.severity}
        </span>
        <span className="text-xs text-muted-foreground font-medium">{issue.category}</span>
        <span className="flex-1 text-sm text-foreground truncate">{issue.title}</span>
        {resolved && <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />}
        {fp && <XCircle className="w-4 h-4 text-muted-foreground shrink-0" />}
        {open ? <ChevronUp className="w-4 h-4 text-muted-foreground shrink-0" /> : <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />}
      </div>
      {open && (
        <div className="px-4 pb-4 border-t border-border pt-3 space-y-3">
          {issue.description && <p className="text-sm text-muted-foreground">{issue.description}</p>}
          {!resolved && !fp && (
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="gap-1.5 text-green-400 border-green-500/30 hover:bg-green-500/10"
                onClick={() => { markResolved(entryId, issue.key); toast.success("Issue marked as resolved"); onAction(); }}>
                <CheckCircle2 className="w-3.5 h-3.5" /> Mark Resolved
              </Button>
              <Button size="sm" variant="outline" className="gap-1.5 text-muted-foreground"
                onClick={() => { markFalsePositive(entryId, issue.key); toast.info("Marked as false positive"); onAction(); }}>
                <XCircle className="w-3.5 h-3.5" /> False Positive
              </Button>
            </div>
          )}
          {resolved && <p className="text-xs text-green-400">Resolved</p>}
          {fp && <p className="text-xs text-muted-foreground">Marked as false positive</p>}
        </div>
      )}
    </div>
  );
}

// ─── Review Card ──────────────────────────────────────────────────────────────

function ReviewCard({ entry, onUpdate }: { entry: HistoryEntry; onUpdate: () => void }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [, setTick] = useState(0);
  const issues = flatIssues(entry);
  const fresh = getEntries().find((e) => e.id === entry.id) ?? entry;
  const refresh = () => { setTick((n) => n + 1); onUpdate(); };

  return (
    <div className={`bg-card border border-border border-l-4 ${SEV_BORDER[entry.severity] ?? SEV_BORDER.none} rounded-xl mb-3 overflow-hidden`}>
      <div className="flex items-center gap-4 p-4 cursor-pointer hover:bg-secondary/20 transition-colors" onClick={() => setOpen((o) => !o)}>
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          <FileCode2 className="w-4 h-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-foreground truncate text-sm">{entry.filename}</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${SEV_CLS[entry.severity] ?? SEV_CLS.none}`}>
              {entry.severity === "none" ? "Clean" : entry.severity}
            </span>
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${STATUS_CLS[entry.status]}`}>
              {entry.status.replace("_", " ")}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-xs text-muted-foreground">
            <Clock className="w-3 h-3" />
            <span>{fmt(entry.submittedAt)}</span>
            <span>·</span>
            <span>{entry.issueCount} issues</span>
          </div>
        </div>
        <ScoreMini score={entry.overallScore} />
        <div className="flex items-center gap-1.5 shrink-0">
          <Button size="sm" variant="outline" className="gap-1.5 text-xs hidden sm:flex h-7"
            onClick={(e) => { e.stopPropagation(); navigate(`/reviews/${entry.id}`); }}>
            <ExternalLink className="w-3 h-3" /> Report
          </Button>
          <Button size="sm" variant="ghost"
            className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 h-7 w-7 p-0"
            onClick={(e) => { e.stopPropagation(); deleteEntry(entry.id); onUpdate(); toast.success("Review deleted"); }}>
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
          {open ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </div>
      </div>

      {open && (
        <div className="px-4 pb-4 border-t border-border pt-3">
          {issues.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-green-400 py-3">
              <CheckCircle2 className="w-4 h-4" /> No issues detected — clean code!
            </div>
          ) : (
            <>
              <p className="text-sm font-medium text-foreground mb-3">
                {issues.length} Issues
                {fresh.resolvedIssues.length > 0 && (
                  <span className="text-green-400 ml-2 font-normal text-xs">({fresh.resolvedIssues.length} resolved)</span>
                )}
              </p>
              {issues.map((iss) => (
                <IssueRow key={iss.key} entryId={entry.id} issue={iss}
                  resolved={fresh.resolvedIssues.includes(iss.key)}
                  fp={fresh.falsePositives.includes(iss.key)}
                  onAction={refresh} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Repo Folder ──────────────────────────────────────────────────────────────

function RepoFolder({
  repoName, entries, defaultOpen, onUpdate, sortBy, severityFilter, statusFilter, search,
}: {
  repoName: string;
  entries: HistoryEntry[];
  defaultOpen: boolean;
  onUpdate: () => void;
  sortBy: "date" | "score" | "issues" | "severity";
  severityFilter: string;
  statusFilter: string;
  search: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const navigate = useNavigate();

  const filtered = entries
    .filter((e) => {
      const matchSearch = !search || e.filename.toLowerCase().includes(search.toLowerCase()) || e.summary.toLowerCase().includes(search.toLowerCase());
      const matchSev = severityFilter === "all" || e.severity === severityFilter;
      const matchStatus = statusFilter === "all" || e.status === statusFilter;
      return matchSearch && matchSev && matchStatus;
    })
    .sort((a, b) => {
      if (sortBy === "score") return a.overallScore - b.overallScore;
      if (sortBy === "issues") return b.issueCount - a.issueCount;
      if (sortBy === "severity") return (SEV_ORDER[a.severity] ?? 5) - (SEV_ORDER[b.severity] ?? 5);
      return new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime();
    });

  const avgScore = entries.length
    ? Math.round(entries.reduce((s, e) => s + e.overallScore, 0) / entries.length)
    : 0;
  const topSev = entries.reduce(
    (best, e) => (SEV_ORDER[e.severity] ?? 5) < (SEV_ORDER[best] ?? 5) ? e.severity : best,
    "none" as string
  );
  const isManual = repoName === "__manual__";
  const displayName = isManual ? "Manual Submissions" : repoName;

  if (filtered.length === 0 && (search || severityFilter !== "all" || statusFilter !== "all")) return null;

  return (
    <div className="mb-4 border border-border rounded-xl overflow-hidden">
      {/* Folder header */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-secondary/20 transition-colors bg-card select-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="text-primary">
          {open
            ? <FolderOpen className="w-5 h-5" />
            : <Folder className="w-5 h-5" />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            {!isManual && <Github className="w-3.5 h-3.5 text-muted-foreground shrink-0" />}
            <span className="font-semibold text-foreground truncate">{displayName}</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${SEV_CLS[topSev] ?? SEV_CLS.none}`}>
              {topSev === "none" ? "Clean" : topSev}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            {entries.length} file{entries.length !== 1 ? "s" : ""} · avg score {avgScore}/100
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {!isManual && (
            <Button size="sm" variant="ghost" className="text-xs text-muted-foreground h-7 gap-1 hidden sm:flex"
              onClick={(e) => { e.stopPropagation(); navigate(`/reviews?repo=${encodeURIComponent(repoName)}`); }}>
              <ExternalLink className="w-3 h-3" /> Filter
            </Button>
          )}
          {open
            ? <ChevronDown className="w-4 h-4 text-muted-foreground" />
            : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
        </div>
      </div>

      {/* Files */}
      {open && (
        <div className="px-4 pt-3 pb-2 bg-background/50 border-t border-border">
          {filtered.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">No files match your filters.</p>
          ) : (
            filtered.map((entry) => <ReviewCard key={entry.id} entry={entry} onUpdate={onUpdate} />)
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

const Reviews = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const repoFilter = searchParams.get("repo") ?? "";

  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [search, setSearch] = useState(() => sessionStorage.getItem("reviews_search") ?? "");
  const [severityFilter, setSeverityFilter] = useState(() => sessionStorage.getItem("reviews_sev") ?? "all");
  const [statusFilter, setStatusFilter] = useState(() => sessionStorage.getItem("reviews_status") ?? "all");
  const [sortBy, setSortBy] = useState<"date" | "score" | "issues" | "severity">(() => (sessionStorage.getItem("reviews_sort") as "date" | "score" | "issues" | "severity") ?? "date");

  useEffect(() => { sessionStorage.setItem("reviews_search", search); }, [search]);
  useEffect(() => { sessionStorage.setItem("reviews_sev", severityFilter); }, [severityFilter]);
  useEffect(() => { sessionStorage.setItem("reviews_status", statusFilter); }, [statusFilter]);
  useEffect(() => { sessionStorage.setItem("reviews_sort", sortBy); }, [sortBy]);

  const reload = () => setEntries(getEntries());
  useEffect(() => {
    reload();
    const onStorage = (e: StorageEvent) => { if (e.key === "intellcode_review_history") reload(); };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // When a repoFilter is active, show only that repo's entries in flat list
  const repoFilteredEntries = useMemo(() => {
    if (!repoFilter) return entries;
    return entries.filter((e) => (e.repoName ?? "") === repoFilter);
  }, [entries, repoFilter]);

  // Group all entries by repo for folder view
  const groupedByRepo = useMemo(() => {
    const map = new Map<string, HistoryEntry[]>();
    entries.forEach((e) => {
      const key = e.repoName ?? "__manual__";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(e);
    });
    // Sort groups: repos alphabetically, manual last
    const sorted = [...map.entries()].sort(([a], [b]) => {
      if (a === "__manual__") return 1;
      if (b === "__manual__") return -1;
      return a.localeCompare(b);
    });
    return sorted;
  }, [entries]);

  // Flat filtered list (used when repoFilter is set)
  const filteredFlat = repoFilteredEntries
    .filter((e) => {
      const matchSearch = !search || e.filename.toLowerCase().includes(search.toLowerCase()) || e.summary.toLowerCase().includes(search.toLowerCase());
      const matchSev = severityFilter === "all" || e.severity === severityFilter;
      const matchStatus = statusFilter === "all" || e.status === statusFilter;
      return matchSearch && matchSev && matchStatus;
    })
    .sort((a, b) => {
      if (sortBy === "score") return a.overallScore - b.overallScore;
      if (sortBy === "issues") return b.issueCount - a.issueCount;
      if (sortBy === "severity") return (SEV_ORDER[a.severity] ?? 5) - (SEV_ORDER[b.severity] ?? 5);
      return new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime();
    });

  const handleExport = () => {
    const data = repoFilter ? repoFilteredEntries : entries;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `intellcode_reviews_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Reviews exported as JSON");
  };

  const handleExportCSV = () => {
    const data = repoFilter ? repoFilteredEntries : entries;
    const header = ["id", "filename", "language", "repoName", "submittedAt", "overallScore", "issueCount", "severity", "status", "summary"];
    const rows = data.map((e) => [
      e.id,
      `"${e.filename.replace(/"/g, '""')}"`,
      e.language,
      `"${(e.repoName ?? "").replace(/"/g, '""')}"`,
      e.submittedAt,
      e.overallScore,
      e.issueCount,
      e.severity,
      e.status,
      `"${(e.summary ?? "").replace(/"/g, '""')}"`,
    ]);
    const csv = [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `intellcode_reviews_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Reviews exported as CSV");
  };

  const handleClear = () => {
    const msg = repoFilter
      ? `Delete all ${repoFilteredEntries.length} reviews for ${repoFilter}? This cannot be undone.`
      : "Delete all review history? This cannot be undone.";
    if (!window.confirm(msg)) return;
    if (repoFilter) {
      repoFilteredEntries.forEach((e) => deleteEntry(e.id));
      reload();
      toast.info("Reviews cleared for this repository");
    } else {
      clearHistory();
      reload();
      toast.info("Review history cleared");
    }
  };

  const displayCount = repoFilter ? repoFilteredEntries.length : entries.length;

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="container mx-auto px-4 py-8">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3 min-w-0">
            {repoFilter && (
              <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground shrink-0 -ml-2"
                onClick={() => setSearchParams({})}>
                <ArrowLeft className="w-4 h-4" /> All
              </Button>
            )}
            <div className="min-w-0">
              {repoFilter ? (
                <>
                  <div className="flex items-center gap-2">
                    <Github className="w-4 h-4 text-muted-foreground shrink-0" />
                    <h1 className="text-xl font-bold text-foreground truncate">{repoFilter}</h1>
                  </div>
                  <p className="text-muted-foreground text-sm mt-0.5">
                    {displayCount} file{displayCount !== 1 ? "s" : ""} analyzed
                  </p>
                </>
              ) : (
                <>
                  <h1 className="text-2xl font-bold text-foreground">Code Reviews</h1>
                  <p className="text-muted-foreground mt-1">
                    {entries.length} analysis{entries.length !== 1 ? "es" : ""} across {groupedByRepo.length} source{groupedByRepo.length !== 1 ? "s" : ""}
                  </p>
                </>
              )}
            </div>
          </div>
          <div className="flex gap-2 shrink-0">
            {displayCount > 0 && (
              <>
                <Button variant="outline" size="sm" onClick={handleExport} className="gap-1.5">
                  <Download className="w-4 h-4" /> JSON
                </Button>
                <Button variant="outline" size="sm" onClick={handleExportCSV} className="gap-1.5">
                  <Download className="w-4 h-4" /> CSV
                </Button>
                <Button variant="outline" size="sm" onClick={handleClear} className="gap-1.5 text-destructive border-destructive/30 hover:bg-destructive/10">
                  <Trash2 className="w-4 h-4" /> Clear
                </Button>
              </>
            )}
            <Button className="bg-gradient-primary gap-1.5" size="sm" onClick={() => navigate("/submit")}>
              + New Analysis
            </Button>
          </div>
        </div>

        {/* Severity summary */}
        {displayCount > 0 && (
          <div className="flex flex-wrap gap-3 mb-5">
            {(["critical", "high", "medium", "low"] as const).map((sev) => {
              const src = repoFilter ? repoFilteredEntries : entries;
              const count = src.filter((e) => e.severity === sev).length;
              if (!count) return null;
              return (
                <div key={sev} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${SEV_CLS[sev]}`}>
                  <AlertTriangle className="w-3.5 h-3.5" />
                  {count} {sev}
                </div>
              );
            })}
          </div>
        )}

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-6">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input placeholder="Search filename or summary…" value={search} onChange={(e) => setSearch(e.target.value)} className="pl-9 bg-input border-border" />
          </div>
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-36 bg-input border-border"><SelectValue placeholder="Severity" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severity</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
              <SelectItem value="none">Clean</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36 bg-input border-border"><SelectValue placeholder="Status" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="in_progress">In Progress</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
            </SelectContent>
          </Select>
          <Select value={sortBy} onValueChange={(v) => setSortBy(v as typeof sortBy)}>
            <SelectTrigger className="w-36 bg-input border-border"><SelectValue placeholder="Sort by" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="date">Newest First</SelectItem>
              <SelectItem value="score">Lowest Score</SelectItem>
              <SelectItem value="issues">Most Issues</SelectItem>
              <SelectItem value="severity">Highest Severity</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Content */}
        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <ShieldAlert className="w-16 h-16 text-muted-foreground/30 mb-4" />
            <h2 className="text-xl font-semibold text-foreground mb-2">No reviews yet</h2>
            <p className="text-muted-foreground mb-6 max-w-sm">Submit your first code snippet and our ML models will analyze it instantly.</p>
            <Button className="bg-gradient-primary" onClick={() => navigate("/submit")}>Submit Code Now</Button>
          </div>
        ) : repoFilter ? (
          /* ── Flat filtered list for a specific repo ── */
          repoFilteredEntries.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Bug className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p>No reviews found for <strong>{repoFilter}</strong>.</p>
              <p className="text-xs mt-1">Scan this repository from the Repositories tab first.</p>
            </div>
          ) : filteredFlat.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Bug className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p>No reviews match your filters.</p>
            </div>
          ) : (
            filteredFlat.map((entry) => <ReviewCard key={entry.id} entry={entry} onUpdate={reload} />)
          )
        ) : (
          /* ── Folder/grouped view ── */
          groupedByRepo.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Bug className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p>No reviews match your filters.</p>
            </div>
          ) : (
            groupedByRepo.map(([key, groupEntries]) => (
              <RepoFolder
                key={key}
                repoName={key}
                entries={groupEntries}
                defaultOpen={groupedByRepo.length === 1}
                onUpdate={reload}
                sortBy={sortBy}
                severityFilter={severityFilter}
                statusFilter={statusFilter}
                search={search}
              />
            ))
          )
        )}
      </main>
    </div>
  );
};

export default Reviews;
