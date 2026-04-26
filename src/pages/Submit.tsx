import { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Github, Loader2, AlertCircle, AlertTriangle, X, FolderOpen, ScanLine, Braces, CheckCircle2, Copy, Check, Zap } from "lucide-react";
import { toast } from "sonner";
import { analyzeCodeStream, analyzeCodeQuick, analyzeBatch } from "@/services/api";
import { addEntry } from "@/services/reviewHistory";
import { authHeaders } from "@/services/github";

const _BASE = import.meta.env.VITE_API_URL ?? "https://intellcode.onrender.com";
async function checkBackend(): Promise<boolean> {
  try {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), 8000);
    const r = await fetch(`${_BASE}/health`, { signal: ac.signal });
    clearTimeout(t);
    return r.ok;
  } catch { return false; }
}

import { STORAGE_KEYS } from "@/constants/storage";

function loadConnectedRepos(): Array<{ id: string; fullName: string; defaultBranch: string; language: string }> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.repositories);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

const EXT_LANG: Record<string, string> = {
  py: "python", js: "javascript", ts: "typescript",
  jsx: "javascript", tsx: "typescript",
  java: "java", go: "go", rs: "rust",
  cpp: "cpp", cc: "cpp", c: "c", cs: "csharp",
  rb: "ruby", php: "php", kt: "kotlin", swift: "swift",
  ipynb: "python",
};

function extractNotebookCode(raw: string): string {
  try {
    const nb = JSON.parse(raw);
    return (nb.cells as Array<{ cell_type: string; source: string | string[] }>)
      .filter((c) => c.cell_type === "code")
      .map((c) => (Array.isArray(c.source) ? c.source.join("") : c.source))
      .filter((src) => src.trim().length > 0)
      .join("\n\n# ── next cell ──\n\n");
  } catch {
    return raw;
  }
}

function detectLanguage(fname: string): string {
  const ext = fname.split(".").pop()?.toLowerCase() ?? "";
  return EXT_LANG[ext] ?? "python";
}

const SAMPLE_CODE = `import os
import json
from typing import List

def process_user_data(users, db_connection):
    results = []
    for user in users:
        query = "SELECT * FROM users WHERE id = " + str(user['id'])
        data = db_connection.execute(query)

        # Calculate metrics
        total = 0
        for i in range(len(data)):
            for j in range(len(data)):
                total += data[i]['value'] * data[j]['weight']

        results.append(total)
    return results

def validate_input(data):
    if data != None:
        if data['type'] == 'A':
            if data['value'] > 0:
                if data['enabled'] == True:
                    return True
    return False

secret_key = "REPLACE_ME_do_not_hardcode"   # ← hardcoded secret (intentional demo issue)
api_token = "REPLACE_ME_use_env_var"        # ← hardcoded token (intentional demo issue)
`;

const SCAN_SKIP = [
  "venv/", "__pycache__/", "node_modules/", ".git/",
  "migrations/", "test_", "_test.py", "conftest.py",
  "setup.py", "checkpoints/", "data/", ".egg-info/",
  "training/", "generate_", "fetch_", "show_metrics",
];

const MODEL_STEPS = [
  { id: "security",     name: "Security Detection" },
  { id: "complexity",   name: "Complexity Analysis" },
  { id: "bugs",         name: "Bug Prediction" },
  { id: "patterns",     name: "Pattern Recognition" },
  { id: "clones",       name: "Clone Detection" },
  { id: "refactoring",  name: "Refactoring Analysis" },
  { id: "deadcode",     name: "Dead Code Detection" },
  { id: "debt",         name: "Technical Debt" },
  { id: "docs",         name: "Documentation Quality" },
  { id: "performance",  name: "Performance Analysis" },
  { id: "deps",         name: "Dependency Analysis" },
  { id: "readability",  name: "Readability Scoring" },
] as const;

const Submit = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const hintFilename = (location.state as { filename?: string } | null)?.filename;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [doneSteps, setDoneSteps] = useState<Set<string>>(new Set());

  // Map SSE step names → MODEL_STEPS ids for real-time highlighting
  const STEP_NAME_TO_ID: Record<string, string> = {
    "Security Detection":   "security",
    "Complexity Analysis":  "complexity",
    "Bug Prediction":       "bugs",
    "Pattern Recognition":  "patterns",
    "Clone Detection":      "clones",
    "Refactoring Analysis": "refactoring",
    "Dead Code Detection":  "deadcode",
    "Technical Debt":       "debt",
    "Documentation Quality": "docs",
    "Performance Analysis": "performance",
    "Dependency Analysis":  "deps",
    "Readability Scoring":  "readability",
  };

  const onStepDone = (stepName: string) => {
    const id = STEP_NAME_TO_ID[stepName];
    if (id) setDoneSteps((prev) => new Set([...prev, id]));
  };

  const startProgress = () => setDoneSteps(new Set());

  const stopProgress = () => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setDoneSteps(new Set(MODEL_STEPS.map((s) => s.id)));
  };
  const [analysisMode, setAnalysisMode] = useState<"full" | "quick">("full");
  const [submissionMethod, setSubmissionMethod] = useState<"upload" | "github">("upload");
  const [connectedRepos, setConnectedRepos] = useState(loadConnectedRepos);
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEYS.repositories) setConnectedRepos(loadConnectedRepos());
    };
    window.addEventListener("storage", onStorage);
    // Reload on tab focus (user coming back from Repositories / Rules page)
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        setConnectedRepos(loadConnectedRepos());
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => { window.removeEventListener("storage", onStorage); document.removeEventListener("visibilitychange", onVisible); };
  }, []);

  // GitHub mode state
  const [ghAuthError, setGhAuthError] = useState(false); // true when token is expired/invalid
  const [selectedRepo, setSelectedRepo] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("main");
  const [branches, setBranches] = useState<string[]>([]);
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [ghFiles, setGhFiles] = useState<Array<{ path: string; size: number }>>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [scanProgress, setScanProgress] = useState<{ done: number; total: number } | null>(null);

  const [code, setCode] = useState(SAMPLE_CODE);
  const [filename, setFilename] = useState(hintFilename ?? "snippet.py");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [codeCopied, setCodeCopied] = useState(false);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code).catch(() => {});
    setCodeCopied(true);
    setTimeout(() => setCodeCopied(false), 2000);
  };
  // Fetch branches when repo changes
  useEffect(() => {
    if (!selectedRepo) return;
    const repo = connectedRepos.find((r) => r.id === selectedRepo);
    if (!repo) return;
    setSelectedBranch(repo.defaultBranch || "main");
    setBranches([]);
    setGhFiles([]);
    setSelectedFile("");
    setLoadingBranches(true);
    fetch(`https://api.github.com/repos/${repo.fullName}/branches`, { headers: authHeaders() })
      .then((r) => {
        if (r.status === 401) throw new Error("GitHub token expired — reconnect in Repositories.");
        if (r.status === 403) throw new Error("GitHub API rate limit reached. Wait a minute and try again.");
        if (r.status === 404) throw new Error("Repository not found or is private.");
        return r.json();
      })
      .then((data: Array<{ name: string }>) => {
        if (!Array.isArray(data)) throw new Error("Unexpected GitHub response");
        setBranches(data.map((b) => b.name));
      })
      .catch((e: Error) => {
        setBranches([repo.defaultBranch || "main"]);
        if (e.message?.includes("token expired") || e.message?.includes("401")) setGhAuthError(true);
        else toast.error(e.message);
      })
      .finally(() => setLoadingBranches(false));
  }, [selectedRepo, connectedRepos]);

  // Fetch file list when branch changes
  useEffect(() => {
    if (!selectedRepo || !selectedBranch) return;
    const repo = connectedRepos.find((r) => r.id === selectedRepo);
    if (!repo) return;
    setGhFiles([]);
    setSelectedFile("");
    setLoadingFiles(true);
    fetch(`https://api.github.com/repos/${repo.fullName}/git/trees/${selectedBranch}?recursive=1`, { headers: authHeaders() })
      .then((r) => {
        if (r.status === 401) throw new Error("GitHub token expired — reconnect in Repositories.");
        if (r.status === 403) throw new Error("GitHub API rate limit reached. Wait a minute and try again.");
        if (r.status === 404) throw new Error("Branch not found.");
        return r.json();
      })
      .then((data: { tree?: Array<{ type: string; path: string; size?: number }> }) => {
        const eligible = (data.tree || []).filter(
          (f) =>
            f.type === "blob" &&
            /\.(py|js|ts|jsx|tsx|java|go|rs|cpp|cc|c|cs|cxx|h|hpp|rb|php|kt|swift|ipynb)$/.test(f.path) &&
            !SCAN_SKIP.some((skip) => f.path.includes(skip))
        );
        // Notebooks can be large (outputs embedded) but we extract code cells only — allow up to 500KB
        const sizeLimit = (f: { path: string }) => f.path.endsWith(".ipynb") ? 500_000 : 100_000;
        const tooLarge = eligible.filter((f) => (f.size ?? 0) >= sizeLimit(f));
        const files = eligible
          .filter((f) => (f.size ?? 0) < sizeLimit(f))
          .map((f) => ({ path: f.path, size: f.size ?? 0 }));
        setGhFiles(files);
        if (files.length > 0) setSelectedFile(files[0].path);
        if (files.length === 0) toast.info("No analyzable files found in this branch.");
        else if (tooLarge.length > 0) toast.info(`${tooLarge.length} file${tooLarge.length !== 1 ? "s" : ""} excluded (>100 KB)`, { description: tooLarge.slice(0, 3).map(f => f.path).join(", ") + (tooLarge.length > 3 ? ` +${tooLarge.length - 3} more` : "") });
      })
      .catch((e: Error) => {
        setGhFiles([]);
        if (e.message?.includes("token expired") || e.message?.includes("401")) setGhAuthError(true);
        else toast.error(e.message);
      })
      .finally(() => setLoadingFiles(false));
  }, [selectedRepo, selectedBranch, connectedRepos]);

  const [isDragOver, setIsDragOver] = useState(false);

  const loadFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setCode(text);
      setFilename(file.name);
      toast.success(`Loaded ${file.name} (${text.split("\n").length} lines)`);
    };
    reader.readAsText(file);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) loadFile(file);
  };

  // Ctrl+Enter → submit
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !isAnalyzing && submissionMethod === "upload") {
        e.preventDefault();
        handleSubmit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAnalyzing, submissionMethod, code, filename]);

  // Paste mode: analyze single file
  const handleSubmit = async () => {
    if (!code.trim()) {
      toast.error("Please paste some code to analyze.");
      return;
    }
    if (code.length > 500_000) {
      toast.error("File too large", { description: "Maximum code size is 500 KB. Please trim or split the file." });
      return;
    }
    if (!await checkBackend()) {
      toast.error("Backend is warming up — wait ~30 seconds and try again");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    startProgress();
    try {
      let result;
      if (analysisMode === "quick") {
        result = await analyzeCodeQuick(code, filename, detectLanguage(filename));
        stopProgress();
      } else {
        result = await analyzeCodeStream(
          code, filename, detectLanguage(filename), onStepDone, undefined, undefined
        );
        stopProgress();
      }
      toast.success("Analysis complete!", { description: result.summary });
      navigate(result._entryId ? `/reviews/${result._entryId}` : "/reviews/result", { state: { result } });
    } catch (err) {
      stopProgress();
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(`Analysis failed: ${msg}. The backend may be starting up — wait 30s and try again.`);
      toast.error("Analysis failed", { description: msg });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // GitHub mode: fetch selected file and analyze it
  const handleGitHubAnalyze = async () => {
    const repo = connectedRepos.find((r) => r.id === selectedRepo);
    if (!repo) { toast.error("Select a repository first"); return; }
    if (!selectedFile) { toast.error("Select a file to analyze"); return; }
    if (!await checkBackend()) {
      toast.error("Backend is warming up — wait ~30 seconds and try again");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setScanProgress(null);

    try {
      toast.info(`Fetching ${selectedFile}…`);
      const rawRes = await fetch(
        `https://raw.githubusercontent.com/${repo.fullName}/${selectedBranch}/${selectedFile}`
      );
      if (!rawRes.ok) throw new Error(`Could not fetch file (${rawRes.status})`);
      const rawCode = await rawRes.text();
      const fileCode = selectedFile.endsWith(".ipynb") ? extractNotebookCode(rawCode) : rawCode;
      if (fileCode.trim().length < 10) throw new Error("File appears to be empty or has no code cells");

      startProgress();
      const result = await analyzeCodeStream(
        fileCode, selectedFile, detectLanguage(selectedFile), onStepDone, undefined, repo.fullName
      );
      stopProgress();
      toast.success("Analysis complete!", { description: result.summary });
      navigate(result._entryId ? `/reviews/${result._entryId}` : "/reviews/result", { state: { result } });
    } catch (err) {
      stopProgress();
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      toast.error("Analysis failed", { description: msg });
    } finally {
      setIsAnalyzing(false);
      setScanProgress(null);
    }
  };

  // GitHub mode: full 12-model scan of all files via /analyze/batch
  // Phase 1 — fetch all raw files concurrently from GitHub CDN (~instant)
  // Phase 2 — send to backend in batches of 8, 3 batches in parallel (24 files at once)
  // Estimated time: ~2 min for 180 files (vs 45 min serial)
  const handleGitHubScanAll = async () => {
    const repo = connectedRepos.find((r) => r.id === selectedRepo);
    if (!repo) { toast.error("Select a repository first"); return; }
    if (ghFiles.length === 0) { toast.error("No analyzable files found"); return; }
    if (!await checkBackend()) {
      toast.error("Backend is warming up — wait ~30 seconds and try again");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setScanProgress({ done: 0, total: ghFiles.length });

    // Phase 1: fetch file contents from GitHub CDN, 20 concurrent
    const FETCH_CONCURRENCY = 20;
    const fetched: Array<{ path: string; code: string; lang: string }> = [];
    for (let i = 0; i < ghFiles.length; i += FETCH_CONCURRENCY) {
      const settled = await Promise.allSettled(
        ghFiles.slice(i, i + FETCH_CONCURRENCY).map(async (file) => {
          const res = await fetch(
            `https://raw.githubusercontent.com/${repo.fullName}/${selectedBranch}/${file.path}`
          );
          if (!res.ok) throw new Error(`${res.status}`);
          const raw = await res.text();
          const code = file.path.endsWith(".ipynb") ? extractNotebookCode(raw) : raw;
          if (code.trim().length < 20) throw new Error("empty");
          return { path: file.path, code, lang: detectLanguage(file.path) };
        })
      );
      for (const r of settled) {
        if (r.status === "fulfilled") fetched.push(r.value);
      }
    }

    if (fetched.length === 0) {
      setIsAnalyzing(false);
      setScanProgress(null);
      toast.error("Could not fetch any files from GitHub");
      return;
    }

    // Phase 2: full analysis via /analyze/batch — 8 files per batch, 3 batches concurrent
    const BATCH_SIZE = 8;
    const BATCH_CONCURRENCY = 3;
    const allResults: Array<{ filename: string }> = [];
    const failedCount = { n: 0 };
    let done = 0;
    setScanProgress({ done: 0, total: fetched.length });

    const batches: typeof fetched[] = [];
    for (let i = 0; i < fetched.length; i += BATCH_SIZE) {
      batches.push(fetched.slice(i, i + BATCH_SIZE));
    }

    for (let i = 0; i < batches.length; i += BATCH_CONCURRENCY) {
      await Promise.all(
        batches.slice(i, i + BATCH_CONCURRENCY).map(async (batchFiles) => {
          try {
            const batchResult = await analyzeBatch(
              batchFiles.map((f) => ({ code: f.code, filename: f.path, language: f.lang }))
            );
            for (const result of batchResult.results) {
              addEntry(result, result.filename, detectLanguage(result.filename), repo.fullName);
              allResults.push(result);
            }
            failedCount.n += batchResult.errors.length;
          } catch {
            failedCount.n += batchFiles.length;
          }
          done += batchFiles.length;
          setScanProgress({ done, total: fetched.length });
        })
      );
    }

    setIsAnalyzing(false);
    setScanProgress(null);

    if (allResults.length === 0) {
      toast.error("No files could be analyzed", {
        description: "The backend may be starting up — wait 30s and try again.",
      });
      return;
    }

    toast.success(`Full scan: ${allResults.length} of ${fetched.length} files analyzed`, {
      description: failedCount.n > 0
        ? `${failedCount.n} failed (backend errors)`
        : "All results saved to review history",
    });
    navigate(repo ? `/reviews?repo=${encodeURIComponent(repo.fullName)}` : "/reviews");
  };

  const repoObj = connectedRepos.find((r) => r.id === selectedRepo);

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-3xl">
        {/* Breadcrumb */}
        <div className="text-sm text-muted-foreground mb-4">
          <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>
            Dashboard
          </span>
          <span className="mx-2">›</span>
          <span className="text-foreground">New Analysis</span>
        </div>

        {/* Hero banner */}
        <div className="relative overflow-hidden rounded-2xl mb-6 border border-primary/20 bg-gradient-to-br from-primary/12 via-primary/6 to-transparent">
          <div className="absolute -top-12 -right-12 w-44 h-44 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
          <div className="relative z-10 flex items-center gap-4 p-6">
            <div className="p-3 bg-primary/15 rounded-xl shrink-0">
              <Braces className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">New Code Analysis</h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                ML models run in parallel — security, bugs, complexity, docs, performance &amp; more.
              </p>
            </div>
          </div>
        </div>

        {hintFilename && (
          <div className="bg-primary/10 border border-primary/30 rounded-xl px-4 py-3 mb-6 text-sm text-primary">
            Re-analyzing <span className="font-mono font-semibold">{hintFilename}</span> — upload or paste the same file below, then click <strong>Run All 12 Models</strong>.
          </div>
        )}

        {/* Submission Method */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Upload className="w-4 h-4 text-primary" />
            <h2 className="text-base font-semibold text-foreground">Submission Method</h2>
          </div>
          <div className="space-y-3">
            <label
              className={`flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all ${
                submissionMethod === "upload"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground"
              }`}
              onClick={() => setSubmissionMethod("upload")}
            >
              <div
                className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                  submissionMethod === "upload" ? "border-primary" : "border-muted-foreground"
                }`}
              >
                {submissionMethod === "upload" && (
                  <div className="w-2 h-2 rounded-full bg-primary" />
                )}
              </div>
              <Upload className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">Paste Code Directly</span>
            </label>
            <label
              className={`flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all ${
                submissionMethod === "github"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground"
              }`}
              onClick={() => setSubmissionMethod("github")}
            >
              <div
                className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                  submissionMethod === "github" ? "border-primary" : "border-muted-foreground"
                }`}
              >
                {submissionMethod === "github" && (
                  <div className="w-2 h-2 rounded-full bg-primary" />
                )}
              </div>
              <Github className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">GitHub Repository</span>
              {connectedRepos.length > 0 && (
                <span className="ml-auto text-xs text-primary font-medium">
                  {connectedRepos.length} connected
                </span>
              )}
            </label>
          </div>
        </div>

        {/* Code Editor */}
        {submissionMethod === "upload" && (
          <div
            className={`bg-card border rounded-xl overflow-hidden mb-6 transition-colors ${isDragOver ? "border-primary bg-primary/5" : "border-border"}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
          >
            {/* Editor chrome bar */}
            <div className="flex items-center justify-between px-4 py-2.5 bg-secondary/60 border-b border-border">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/70" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                  <div className="w-3 h-3 rounded-full bg-green-500/70" />
                </div>
                <input
                  value={filename}
                  onChange={(e) => setFilename(e.target.value)}
                  placeholder="snippet.py"
                  className="ml-2 bg-transparent font-mono text-sm text-foreground focus:outline-none border-none w-40"
                />
                <span className="text-xs text-muted-foreground bg-secondary px-1.5 py-0.5 rounded capitalize">
                  {detectLanguage(filename)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground tabular-nums">
                  {code.split("\n").length}L · {code.length}c
                </span>
                <Button variant="ghost" size="sm" className="gap-1.5 h-7 text-xs" onClick={handleCopyCode}>
                  {codeCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {codeCopied ? "Copied!" : "Copy"}
                </Button>
                <Button variant="outline" size="sm" className="gap-1.5 h-7 text-xs" onClick={() => fileInputRef.current?.click()}>
                  <FolderOpen className="w-3.5 h-3.5" /> Open
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.cpp,.cc,.c,.cs,.rb,.php,.kt,.swift,.txt"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </div>
            </div>
            <div className="relative">
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                rows={22}
                spellCheck={false}
                className="w-full bg-[hsl(220,26%,4%)] font-mono text-sm text-foreground p-4 resize-y focus:outline-none border-none"
                placeholder="Paste your code here, or drag & drop a file..."
              />
              {isDragOver && (
                <div className="absolute inset-0 flex items-center justify-center bg-primary/10 border-2 border-dashed border-primary rounded pointer-events-none">
                  <p className="text-primary font-semibold text-sm">Drop file to load</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* GitHub integration */}
        {submissionMethod === "github" && (
          <div className="bg-card border border-border rounded-xl p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Github className="w-4 h-4 text-primary" />
              <h2 className="text-base font-semibold text-foreground">Repository Selection</h2>
            </div>
            {ghAuthError && (
              <div className="flex items-start gap-3 mb-4 px-4 py-3 bg-destructive/10 border border-destructive/30 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-destructive">GitHub token expired</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Your access token is no longer valid.{" "}
                    <span
                      className="text-primary underline cursor-pointer"
                      onClick={() => navigate("/repositories", { state: { from: "submit" } })}
                    >
                      Reconnect in Repositories
                    </span>{" "}
                    to continue.
                  </p>
                </div>
                <button type="button" onClick={() => setGhAuthError(false)} className="text-muted-foreground hover:text-foreground">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
            {connectedRepos.length === 0 ? (
              <div className="flex flex-col items-center gap-3 py-6 text-center">
                <Github className="w-10 h-10 text-muted-foreground opacity-50" />
                <p className="text-sm text-muted-foreground">
                  No repositories connected yet.
                </p>
                <Button
                  size="sm"
                  className="bg-gradient-primary gap-2"
                  onClick={() => navigate("/repositories", { state: { from: "submit" } })}
                >
                  <Github className="w-4 h-4" />
                  Connect a Repository
                </Button>
                <p className="text-xs text-muted-foreground">
                  Or{" "}
                  <span
                    className="text-primary cursor-pointer underline"
                    onClick={() => setSubmissionMethod("upload")}
                  >
                    paste code directly
                  </span>{" "}
                  to analyze now.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <Label className="text-foreground mb-2 block">Repository</Label>
                  <Select value={selectedRepo} onValueChange={setSelectedRepo}>
                    <SelectTrigger className="bg-input border-border">
                      <SelectValue placeholder="Select repository" />
                    </SelectTrigger>
                    <SelectContent>
                      {connectedRepos.map((repo) => (
                        <SelectItem key={repo.id} value={repo.id}>
                          {repo.fullName}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {selectedRepo && (
                  <div>
                    <Label className="text-foreground mb-2 block">Branch</Label>
                    <Select value={selectedBranch} onValueChange={setSelectedBranch} disabled={loadingBranches}>
                      <SelectTrigger className="bg-input border-border">
                        <SelectValue placeholder={loadingBranches ? "Loading…" : "Select branch"} />
                      </SelectTrigger>
                      <SelectContent>
                        {branches.length > 0
                          ? branches.map((b) => <SelectItem key={b} value={b}>{b}</SelectItem>)
                          : <SelectItem value={selectedBranch}>{selectedBranch}</SelectItem>
                        }
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {selectedRepo && (
                  <div>
                    <Label className="text-foreground mb-2 block">
                      File to Analyze
                      {loadingFiles && <span className="ml-2 text-xs text-muted-foreground">Loading files…</span>}
                      {!loadingFiles && ghFiles.length > 0 && (
                        <span className="ml-2 text-xs text-muted-foreground">
                          {ghFiles.length} files
                        </span>
                      )}
                    </Label>
                    {ghFiles.length > 0 ? (
                      <Select value={selectedFile} onValueChange={setSelectedFile}>
                        <SelectTrigger className="bg-input border-border font-mono text-xs">
                          <SelectValue placeholder="Select file" />
                        </SelectTrigger>
                        <SelectContent>
                          {ghFiles.map((f) => (
                            <SelectItem key={f.path} value={f.path} className="font-mono text-xs">
                              {f.path}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : !loadingFiles ? (
                      <p className="text-sm text-muted-foreground">
                        No analyzable files found in this branch.
                      </p>
                    ) : null}
                  </div>
                )}

                {scanProgress && (
                  <div>
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>Scanning files…</span>
                      <span>{scanProgress.done}/{scanProgress.total}</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${(scanProgress.done / scanProgress.total) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {selectedRepo && ghFiles.length > 0 && (
                  <div className="flex gap-2 pt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1.5"
                      onClick={handleGitHubScanAll}
                      disabled={isAnalyzing}
                    >
                      <ScanLine className="w-4 h-4" />
                      Full Scan All {ghFiles.length} Files
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Analysis mode toggle */}
        {submissionMethod === "upload" && (
          <div className="bg-card border border-border rounded-xl p-4 mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-primary" />
              <h2 className="text-sm font-semibold text-foreground">Analysis Mode</h2>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setAnalysisMode("full")}
                className={`p-3 rounded-xl border text-left transition-all ${
                  analysisMode === "full"
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-muted-foreground"
                }`}
              >
                <div className="font-medium text-sm text-foreground mb-0.5">Full Analysis</div>
                <div className="text-xs text-muted-foreground">All 12 ML models · ~15s</div>
              </button>
              <button
                onClick={() => setAnalysisMode("quick")}
                className={`p-3 rounded-xl border text-left transition-all ${
                  analysisMode === "quick"
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-muted-foreground"
                }`}
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="font-medium text-sm text-foreground">Quick Scan</span>
                  <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded font-semibold">FAST</span>
                </div>
                <div className="text-xs text-muted-foreground">Security · Complexity · Readability · ~3s</div>
              </button>
            </div>
          </div>
        )}


        {/* Model progress */}
        {isAnalyzing && (
          <div className="bg-card border border-border rounded-xl p-4 mb-6">
            <p className="text-xs font-semibold text-muted-foreground mb-3 flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
              Running models…
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {MODEL_STEPS.map((step) => {
                const done = doneSteps.has(step.id);
                return (
                  <div
                    key={step.id}
                    className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs transition-all duration-300 ${
                      done
                        ? "bg-primary/10 text-primary"
                        : "bg-secondary/40 text-muted-foreground"
                    }`}
                  >
                    {done ? (
                      <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                    ) : (
                      <div className="w-3.5 h-3.5 rounded-full border-2 border-current shrink-0 opacity-40" />
                    )}
                    {step.name}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-destructive/10 border border-destructive/30 rounded-xl p-4 mb-6 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-4">
          <Button
            variant="outline"
            onClick={() => navigate("/dashboard")}
            disabled={isAnalyzing}
          >
            Cancel
          </Button>
          {submissionMethod === "github" ? (
            <Button
              className="bg-gradient-primary min-w-[180px]"
              onClick={handleGitHubAnalyze}
              disabled={isAnalyzing || !selectedRepo || !selectedFile}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {scanProgress ? `${scanProgress.done}/${scanProgress.total} files…` : "Analyzing…"}
                </>
              ) : (
                `Analyze ${selectedFile ? selectedFile.split("/").pop() : "File"} →`
              )}
            </Button>
          ) : (
            <Button
              className="bg-gradient-primary min-w-[180px]"
              onClick={handleSubmit}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : analysisMode === "quick" ? (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Quick Scan →
                  <span className="ml-2 text-xs opacity-60 hidden sm:inline">Ctrl+↵</span>
                </>
              ) : (
                <>
                  Run All 12 Models →
                  <span className="ml-2 text-xs opacity-60 hidden sm:inline">Ctrl+↵</span>
                </>
              )}
            </Button>
          )}
        </div>
      </main>
    </div>
  );
};

export default Submit;
