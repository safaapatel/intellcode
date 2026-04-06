import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import {
  Upload, X, Play, CheckCircle2, XCircle, AlertTriangle, FileCode2,
  BarChart3, Download, ExternalLink, Loader2, Shield, Zap,
} from "lucide-react";
import { toast } from "sonner";
import { analyzeBatch, type BatchFileInput, type BatchAnalysisResult } from "@/services/api";
import { addEntry } from "@/services/reviewHistory";

// ─── Constants ────────────────────────────────────────────────────────────────

const EXT_LANG: Record<string, string> = {
  py: "python", js: "javascript", ts: "typescript",
  jsx: "javascript", tsx: "typescript", java: "java",
  go: "go", rs: "rust", cpp: "cpp", cc: "cpp", c: "c",
  cs: "csharp", rb: "ruby", php: "php", kt: "kotlin", swift: "swift",
};

function detectLanguage(fname: string): string {
  const ext = fname.split(".").pop()?.toLowerCase() ?? "";
  return EXT_LANG[ext] ?? "python";
}

const SEV_CLS: Record<string, string> = {
  critical: "bg-red-600 text-white",
  high:     "bg-orange-500 text-white",
  medium:   "bg-yellow-500 text-black",
  low:      "bg-blue-500 text-white",
  none:     "bg-emerald-600 text-white",
};

const SEV_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, none: 4 };

function topSeverityFromResult(r: BatchAnalysisResult["results"][number]): string {
  const findings = r.security?.findings ?? [];
  if (findings.some((f) => f.severity === "critical")) return "critical";
  if (r.bug_prediction?.risk_level === "critical") return "critical";
  if (findings.some((f) => f.severity === "high")) return "high";
  if (r.bug_prediction?.risk_level === "high") return "high";
  if (findings.some((f) => f.severity === "medium")) return "medium";
  if (findings.length > 0) return "low";
  return "none";
}

// ─── Queued file ──────────────────────────────────────────────────────────────

interface QueuedFile {
  id: string;
  name: string;
  language: string;
  code: string;
  size: number;
  status: "pending" | "analyzing" | "done" | "error";
  score?: number;
  severity?: string;
  issueCount?: number;
  entryId?: string;
  error?: string;
}

// ─── Score badge ──────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number }) {
  const color = score >= 80 ? "bg-emerald-500" : score >= 60 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-secondary rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs font-semibold tabular-nums w-8 text-right text-foreground">{score}/100</span>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

const MAX_FILES = 20;

const Batch = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<BatchAnalysisResult | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const addFiles = useCallback((files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) =>
      /\.(py|js|ts|jsx|tsx|java|go|rs|cpp|cc|c|cs|rb|php|kt|swift)$/.test(f.name)
    );
    if (arr.length === 0) {
      toast.error("No supported files found", { description: "Supported: .py .js .ts .jsx .tsx .java .go .rs .cpp .rb .php" });
      return;
    }
    const remaining = MAX_FILES - queue.length;
    if (remaining <= 0) {
      toast.error(`Maximum ${MAX_FILES} files`);
      return;
    }
    const toAdd = arr.slice(0, remaining);
    if (arr.length > remaining) {
      toast.info(`Added ${toAdd.length} files (limit: ${MAX_FILES})`);
    }

    Promise.all(
      toAdd.map(
        (f) =>
          new Promise<QueuedFile>((resolve) => {
            const reader = new FileReader();
            reader.onload = (ev) => {
              resolve({
                id: crypto.randomUUID(),
                name: f.name,
                language: detectLanguage(f.name),
                code: ev.target?.result as string,
                size: f.size,
                status: "pending",
              });
            };
            reader.readAsText(f);
          })
      )
    ).then((items) => {
      setQueue((prev) => [...prev, ...items]);
      setResults(null);
    });
  }, [queue.length]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const removeFile = (id: string) => {
    setQueue((prev) => prev.filter((f) => f.id !== id));
    setResults(null);
  };

  const handleAnalyze = async () => {
    if (queue.length === 0) return;
    setIsAnalyzing(true);
    setResults(null);
    setQueue((prev) => prev.map((f) => ({ ...f, status: "analyzing" as const })));

    try {
      const inputs: BatchFileInput[] = queue.map((f) => ({
        code: f.code,
        filename: f.name,
        language: f.language,
      }));

      const batchResult = await analyzeBatch(inputs);
      setResults(batchResult);

      // Persist each result individually to history
      const savedIds: Record<string, string> = {};
      batchResult.results.forEach((r) => {
        const entry = addEntry(r, r.filename, detectLanguage(r.filename));
        savedIds[r.filename] = entry.id;
      });

      // Update queue with results
      setQueue((prev) =>
        prev.map((f) => {
          const match = batchResult.results.find((r) => r.filename === f.name);
          if (match) {
            const score = match.overall_score ?? 0;
            const sev = topSeverityFromResult(match);
            const issueCount =
              (match.security?.findings?.length ?? 0) +
              (match.dead_code?.issues?.length ?? 0) +
              (match.refactoring?.suggestions?.length ?? 0) +
              (match.performance?.issues?.length ?? 0);
            return {
              ...f,
              status: "done" as const,
              score,
              severity: sev,
              issueCount,
              entryId: savedIds[f.name],
            };
          }
          const err = batchResult.errors.find((_, i) => batchResult.results.length + i < inputs.length);
          return { ...f, status: "error" as const, error: err?.error ?? "Analysis failed" };
        })
      );

      toast.success(`Batch complete — ${batchResult.aggregate.files_analysed} files analysed`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Batch failed";
      toast.error(msg);
      setQueue((prev) => prev.map((f) => ({ ...f, status: "error" as const, error: msg })));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExportCSV = () => {
    if (!results) return;
    const header = ["filename", "score", "severity", "security_findings", "avg_score"];
    const rows = results.results.map((r) => [
      `"${r.filename}"`,
      r.overall_score ?? 0,
      topSeverityFromResult(r),
      r.security?.findings?.length ?? 0,
      results.aggregate.avg_score.toFixed(1),
    ]);
    const csv = [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `intellcode_batch_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Exported as CSV");
  };

  const pendingCount = queue.filter((f) => f.status === "pending").length;
  const doneCount = queue.filter((f) => f.status === "done").length;
  const errorCount = queue.filter((f) => f.status === "error").length;

  // Sort done files: worst first
  const displayQueue = [...queue].sort((a, b) => {
    if (a.status !== b.status) {
      const ord: Record<string, number> = { error: 0, done: 1, analyzing: 2, pending: 3 };
      return (ord[a.status] ?? 4) - (ord[b.status] ?? 4);
    }
    if (a.severity && b.severity) {
      return (SEV_ORDER[a.severity] ?? 5) - (SEV_ORDER[b.severity] ?? 5);
    }
    return 0;
  });

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="container mx-auto px-4 py-8 max-w-5xl">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Batch Analysis</h1>
          <p className="text-muted-foreground mt-2">
            Upload up to {MAX_FILES} source files and analyze them all at once with IntelliCode's ML models.
          </p>
        </div>

        {/* Aggregate stats (shown after results) */}
        {results && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
            {[
              { label: "Files Analysed", value: results.aggregate.files_analysed, icon: FileCode2, color: "text-primary" },
              { label: "Avg Score", value: `${results.aggregate.avg_score.toFixed(1)}/100`, icon: BarChart3, color: results.aggregate.avg_score >= 70 ? "text-emerald-400" : results.aggregate.avg_score >= 50 ? "text-yellow-400" : "text-red-400" },
              { label: "Security Findings", value: results.aggregate.total_security_findings, icon: Shield, color: results.aggregate.total_security_findings > 0 ? "text-orange-400" : "text-emerald-400" },
              { label: "Critical Files", value: results.aggregate.critical_files.length, icon: AlertTriangle, color: results.aggregate.critical_files.length > 0 ? "text-red-400" : "text-emerald-400" },
            ].map(({ label, value, icon: Icon, color }) => (
              <div key={label} className="bg-card border border-border rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Icon className={`w-4 h-4 ${color}`} />
                  <span className="text-xs text-muted-foreground font-medium">{label}</span>
                </div>
                <div className={`text-2xl font-bold ${color}`}>{value}</div>
              </div>
            ))}
          </div>
        )}

        {/* Drop zone */}
        {queue.length < MAX_FILES && (
          <div
            className={`border-2 border-dashed rounded-xl p-8 mb-6 text-center transition-all cursor-pointer ${
              isDragOver
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50 hover:bg-secondary/30"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.cpp,.cc,.c,.cs,.rb,.php,.kt,.swift"
              className="hidden"
              onChange={(e) => e.target.files && addFiles(e.target.files)}
            />
            <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
            <p className="text-foreground font-medium mb-1">
              {isDragOver ? "Drop files here" : "Drag & drop files or click to browse"}
            </p>
            <p className="text-xs text-muted-foreground">
              Supports .py .js .ts .jsx .tsx .java .go .rs .cpp .rb .php · Max {MAX_FILES} files
            </p>
            {queue.length > 0 && (
              <p className="text-xs text-primary mt-2 font-medium">
                {queue.length} file{queue.length !== 1 ? "s" : ""} queued · {MAX_FILES - queue.length} remaining
              </p>
            )}
          </div>
        )}

        {/* Controls */}
        {queue.length > 0 && (
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <span>{queue.length} file{queue.length !== 1 ? "s" : ""}</span>
              {doneCount > 0 && <span className="text-emerald-400">{doneCount} done</span>}
              {errorCount > 0 && <span className="text-red-400">{errorCount} failed</span>}
            </div>
            <div className="flex gap-2">
              {results && (
                <Button variant="outline" size="sm" onClick={handleExportCSV} className="gap-1.5">
                  <Download className="w-4 h-4" /> Export CSV
                </Button>
              )}
              <Button variant="outline" size="sm" onClick={() => { setQueue([]); setResults(null); }}
                className="gap-1.5 text-destructive border-destructive/30 hover:bg-destructive/10">
                <X className="w-4 h-4" /> Clear All
              </Button>
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing || pendingCount === 0}
                className="bg-gradient-primary gap-1.5"
                size="sm"
              >
                {isAnalyzing
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                  : <><Play className="w-4 h-4" /> Analyze {pendingCount > 0 ? pendingCount : queue.length} File{(pendingCount || queue.length) !== 1 ? "s" : ""}</>}
              </Button>
            </div>
          </div>
        )}

        {/* File table */}
        {queue.length > 0 && (
          <div className="space-y-2">
            {displayQueue.map((f) => (
              <div key={f.id}
                className={`bg-card border border-border rounded-xl px-4 py-3 flex items-center gap-4 transition-all ${
                  f.status === "done" && f.severity === "critical" ? "border-l-4 border-l-red-500" :
                  f.status === "done" && f.severity === "high"     ? "border-l-4 border-l-orange-500" :
                  f.status === "done" && f.severity === "none"     ? "border-l-4 border-l-emerald-500" :
                  f.status === "error"                             ? "border-l-4 border-l-red-500 opacity-70" : ""
                }`}>

                {/* Icon */}
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  {f.status === "analyzing" ? (
                    <Loader2 className="w-4 h-4 text-primary animate-spin" />
                  ) : f.status === "done" ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : f.status === "error" ? (
                    <XCircle className="w-4 h-4 text-red-400" />
                  ) : (
                    <FileCode2 className="w-4 h-4 text-primary" />
                  )}
                </div>

                {/* File info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm text-foreground truncate">{f.name}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground font-mono">
                      {f.language}
                    </span>
                    {f.status === "done" && f.severity && (
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${SEV_CLS[f.severity] ?? SEV_CLS.none}`}>
                        {f.severity === "none" ? "Clean" : f.severity}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
                    <span>{(f.size / 1024).toFixed(1)} KB</span>
                    {f.status === "analyzing" && <span className="text-primary animate-pulse">Analysing…</span>}
                    {f.status === "error" && <span className="text-red-400">{f.error}</span>}
                    {f.status === "done" && f.issueCount !== undefined && (
                      <span>{f.issueCount} issue{f.issueCount !== 1 ? "s" : ""}</span>
                    )}
                  </div>
                </div>

                {/* Score */}
                {f.status === "done" && f.score !== undefined && (
                  <ScoreBar score={f.score} />
                )}

                {/* Actions */}
                <div className="flex items-center gap-1.5 shrink-0">
                  {f.status === "done" && f.entryId && (
                    <Button size="sm" variant="outline" className="gap-1.5 h-7 text-xs"
                      onClick={() => navigate(`/reviews/${f.entryId}`)}>
                      <ExternalLink className="w-3 h-3" /> Report
                    </Button>
                  )}
                  {!isAnalyzing && (
                    <button
                      onClick={() => removeFile(f.id)}
                      className="p-1.5 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {queue.length === 0 && (
          <div className="text-center py-16">
            <Zap className="w-16 h-16 text-muted-foreground/20 mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-foreground mb-2">Ready for batch analysis</h2>
            <p className="text-muted-foreground text-sm max-w-sm mx-auto mb-6">
              Upload multiple source files to analyze them all in one request — get aggregate stats and per-file results.
            </p>
            <Button className="bg-gradient-primary gap-2" onClick={() => fileInputRef.current?.click()}>
              <Upload className="w-4 h-4" /> Upload Files
            </Button>
          </div>
        )}

        {/* Critical files callout */}
        {results && results.aggregate.critical_files.length > 0 && (
          <div className="mt-6 border border-red-500/30 bg-red-500/5 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="text-sm font-semibold text-red-400">Critical Files Require Attention</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {results.aggregate.critical_files.map((name) => (
                <span key={name} className="text-xs font-mono bg-red-500/10 border border-red-500/20 text-red-300 px-2 py-1 rounded">
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Batch;
