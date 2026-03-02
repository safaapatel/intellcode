import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
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
  CheckCircle2,
  XCircle,
  ExternalLink,
  AlertTriangle,
  ShieldAlert,
  Bug,
  Trash2,
  Clock,
  FileCode2,
} from "lucide-react";
import { toast } from "sonner";
import {
  getEntries,
  markResolved,
  markFalsePositive,
  clearHistory,
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

const STATUS_CLS: Record<string, string> = {
  completed:   "bg-green-500/20 text-green-400 border-green-500/30",
  in_progress: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  pending:     "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Flatten all issues from a FullAnalysisResult into a stable list */
function flatIssues(entry: HistoryEntry) {
  const r = entry.result;
  const items: { key: string; severity: string; category: string; title: string; description: string }[] = [];

  (r.security?.findings ?? []).forEach((f, i) =>
    items.push({ key: `security_${i}`, severity: f.severity ?? "medium", category: "Security", title: f.title ?? f.vuln_type ?? "Security Issue", description: f.description ?? "" })
  );
  if (r.bugs?.risk_level && r.bugs.risk_level !== "low") {
    items.push({ key: "bug_risk", severity: r.bugs.risk_level === "critical" ? "critical" : r.bugs.risk_level === "high" ? "high" : "medium", category: "Bug Risk", title: `Bug probability: ${Math.round((r.bugs.bug_probability ?? 0) * 100)}% (${r.bugs.risk_level})`, description: (r.bugs.risk_factors ?? []).join("; ") });
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
  const [tick, setTick] = useState(0);
  const issues = flatIssues(entry);
  const fresh = getEntries().find((e) => e.id === entry.id) ?? entry;

  const refresh = () => { setTick((n) => n + 1); onUpdate(); };

  return (
    <div className="bg-card border border-border rounded-xl mb-4 overflow-hidden">
      <div className="flex items-center gap-4 p-5 cursor-pointer hover:bg-secondary/20 transition-colors" onClick={() => setOpen((o) => !o)}>
        <FileCode2 className="w-8 h-8 text-primary shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-foreground truncate">{entry.filename}</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${SEV_CLS[entry.severity] ?? SEV_CLS.none}`}>
              {entry.severity === "none" ? "Clean" : entry.severity}
            </span>
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${STATUS_CLS[entry.status]}`}>
              {entry.status.replace("_", " ")}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            <Clock className="w-3.5 h-3.5" />
            <span>{fmt(entry.submittedAt)}</span>
            <span>·</span>
            <span>{entry.issueCount} issues</span>
            <span>·</span>
            <span>Score {entry.overallScore}/100</span>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button size="sm" variant="outline" className="gap-1.5 text-xs"
            onClick={(e) => { e.stopPropagation(); navigate("/reviews/result", { state: { result: entry.result } }); }}>
            <ExternalLink className="w-3.5 h-3.5" /> Full Report
          </Button>
          {open ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </div>
      </div>

      {open && (
        <div className="px-5 pb-5 border-t border-border pt-4">
          {issues.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-green-400 py-4">
              <CheckCircle2 className="w-5 h-5" /> No issues detected — clean code!
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

// ─── Main Page ────────────────────────────────────────────────────────────────

const Reviews = () => {
  const navigate = useNavigate();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  const reload = () => setEntries(getEntries());
  useEffect(() => { reload(); }, []);

  const filtered = entries.filter((e) => {
    const matchSearch = !search || e.filename.toLowerCase().includes(search.toLowerCase()) || e.summary.toLowerCase().includes(search.toLowerCase());
    const matchSev = severityFilter === "all" || e.severity === severityFilter;
    const matchStatus = statusFilter === "all" || e.status === statusFilter;
    return matchSearch && matchSev && matchStatus;
  });

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `intellcode_reviews_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Reviews exported as JSON");
  };

  const handleClear = () => {
    if (window.confirm("Delete all review history? This cannot be undone.")) {
      clearHistory();
      reload();
      toast.info("Review history cleared");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Code Reviews</h1>
            <p className="text-muted-foreground mt-1">{entries.length} analysis{entries.length !== 1 ? "es" : ""} in history</p>
          </div>
          <div className="flex gap-2">
            {entries.length > 0 && (
              <>
                <Button variant="outline" size="sm" onClick={handleExport} className="gap-1.5">
                  <Download className="w-4 h-4" /> Export
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
        {entries.length > 0 && (
          <div className="flex flex-wrap gap-3 mb-6">
            {(["critical", "high", "medium", "low"] as const).map((sev) => {
              const count = entries.filter((e) => e.severity === sev).length;
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
            <Input placeholder="Search filename or summary..." value={search} onChange={(e) => setSearch(e.target.value)} className="pl-9 bg-input border-border" />
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
            </SelectContent>
          </Select>
        </div>

        {/* List */}
        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <ShieldAlert className="w-16 h-16 text-muted-foreground/30 mb-4" />
            <h2 className="text-xl font-semibold text-foreground mb-2">No reviews yet</h2>
            <p className="text-muted-foreground mb-6 max-w-sm">Submit your first code snippet and all 12 ML models will analyze it instantly.</p>
            <Button className="bg-gradient-primary" onClick={() => navigate("/submit")}>Submit Code Now</Button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Bug className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p>No reviews match your filters.</p>
          </div>
        ) : (
          filtered.map((entry) => <ReviewCard key={entry.id} entry={entry} onUpdate={reload} />)
        )}
      </main>
    </div>
  );
};

export default Reviews;
