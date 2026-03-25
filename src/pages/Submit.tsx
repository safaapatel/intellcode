import { useState, useEffect, useRef } from "react";
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
import { Upload, Github, Loader2, AlertCircle, FolderOpen, ScanLine, Braces, Settings2, CheckCircle2, Copy, Check } from "lucide-react";
import { mockRuleSets } from "@/data/mockData";
import { toast } from "sonner";
import { analyzeCode, analyzeCodeStream } from "@/services/api";

function loadRuleSets() {
  try {
    const raw = localStorage.getItem("intellcode_rules");
    const saved = raw ? JSON.parse(raw) : null;
    return saved?.ruleSets ?? mockRuleSets;
  } catch {
    return mockRuleSets;
  }
}

function loadConnectedRepos(): Array<{ id: string; fullName: string; defaultBranch: string; language: string }> {
  try {
    const raw = localStorage.getItem("intellcode_repositories");
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
};

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

secret_key = "hardcoded-secret-key-12345"
api_token = "sk-prod-abc123xyz"
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
  const [submissionMethod, setSubmissionMethod] = useState<"upload" | "github">("upload");
  const ruleSets = loadRuleSets();
  const [connectedRepos] = useState(loadConnectedRepos);

  // GitHub mode state
  const [selectedRepo, setSelectedRepo] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("main");
  const [branches, setBranches] = useState<string[]>([]);
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [ghFiles, setGhFiles] = useState<Array<{ path: string; size: number }>>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [scanProgress, setScanProgress] = useState<{ done: number; total: number } | null>(null);

  const [selectedRuleSet, setSelectedRuleSet] = useState("");
  const [priority, setPriority] = useState<"normal" | "high" | "critical">("normal");
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
  const [options, setOptions] = useState({
    securityScan: true,
    codeSmell: true,
    mlSuggestions: true,
    refactoring: true,
  });

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
    fetch(`https://api.github.com/repos/${repo.fullName}/branches`)
      .then((r) => {
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
        if (e.message) toast.error(e.message);
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
    fetch(`https://api.github.com/repos/${repo.fullName}/git/trees/${selectedBranch}?recursive=1`)
      .then((r) => {
        if (r.status === 403) throw new Error("GitHub API rate limit reached. Wait a minute and try again.");
        if (r.status === 404) throw new Error("Branch not found.");
        return r.json();
      })
      .then((data: { tree?: Array<{ type: string; path: string; size?: number }> }) => {
        const files = (data.tree || [])
          .filter(
            (f) =>
              f.type === "blob" &&
              /\.(py|js|ts|jsx|tsx|java|go|rs|cpp|cc|c|cs|rb|php|kt|swift)$/.test(f.path) &&
              (f.size ?? 0) < 100_000 &&
              !SCAN_SKIP.some((skip) => f.path.includes(skip))
          )
          .slice(0, 30)
          .map((f) => ({ path: f.path, size: f.size ?? 0 }));
        setGhFiles(files);
        if (files.length > 0) setSelectedFile(files[0].path);
        if (files.length === 0) toast.info("No analyzable files found in this branch.");
      })
      .catch((e: Error) => {
        setGhFiles([]);
        if (e.message) toast.error(e.message);
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
    setIsAnalyzing(true);
    setError(null);
    startProgress();
    try {
      const result = await analyzeCodeStream(
        code, filename, detectLanguage(filename), onStepDone
      );
      stopProgress();
      toast.success("Analysis complete!", { description: result.summary });
      navigate(result._entryId ? `/reviews/${result._entryId}` : "/reviews/result", { state: { result } });
    } catch (err) {
      stopProgress();
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(`Backend error: ${msg}. Make sure the Python backend is running on localhost:8000.`);
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

    setIsAnalyzing(true);
    setError(null);
    setScanProgress(null);

    try {
      toast.info(`Fetching ${selectedFile}…`);
      const rawRes = await fetch(
        `https://raw.githubusercontent.com/${repo.fullName}/${selectedBranch}/${selectedFile}`
      );
      if (!rawRes.ok) throw new Error(`Could not fetch file (${rawRes.status})`);
      const fileCode = await rawRes.text();
      if (fileCode.trim().length < 10) throw new Error("File appears to be empty");

      startProgress();
      const result = await analyzeCodeStream(
        fileCode, selectedFile, detectLanguage(selectedFile), onStepDone
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

  // GitHub mode: scan ALL files in repo
  const handleGitHubScanAll = async () => {
    const repo = connectedRepos.find((r) => r.id === selectedRepo);
    if (!repo) { toast.error("Select a repository first"); return; }
    if (ghFiles.length === 0) { toast.error("No analyzable files found"); return; }

    setIsAnalyzing(true);
    setError(null);
    setScanProgress({ done: 0, total: ghFiles.length });

    const results: Awaited<ReturnType<typeof analyzeCode>>[] = [];
    for (let i = 0; i < ghFiles.length; i++) {
      const file = ghFiles[i];
      try {
        const rawRes = await fetch(
          `https://raw.githubusercontent.com/${repo.fullName}/${selectedBranch}/${file.path}`
        );
        if (!rawRes.ok) continue;
        const fileCode = await rawRes.text();
        if (fileCode.trim().length < 20) continue;
        const result = await analyzeCode(fileCode, file.path, detectLanguage(file.path));
        results.push(result);
      } catch {
        // skip individual failures
      }
      setScanProgress({ done: i + 1, total: ghFiles.length });
    }

    setIsAnalyzing(false);
    setScanProgress(null);

    if (results.length === 0) {
      toast.error("No files could be analyzed");
      return;
    }

    // Navigate to Reviews list so all scanned files are visible
    toast.success(`Scanned ${results.length} file${results.length !== 1 ? "s" : ""}`, {
      description: "All results saved to review history",
    });
    navigate("/reviews");
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
                12 ML models run in parallel — security, bugs, complexity, docs, performance &amp; more.
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
            {connectedRepos.length === 0 ? (
              <div className="flex flex-col items-center gap-3 py-6 text-center">
                <Github className="w-10 h-10 text-muted-foreground opacity-50" />
                <p className="text-sm text-muted-foreground">
                  No repositories connected yet.
                </p>
                <Button
                  size="sm"
                  className="bg-gradient-primary gap-2"
                  onClick={() => navigate("/repositories")}
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
                          {ghFiles.length} files{ghFiles.length === 30 ? " (first 30 shown)" : ""}
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
                      Scan All {ghFiles.length} Files
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Configuration */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Settings2 className="w-4 h-4 text-primary" />
            <h2 className="text-base font-semibold text-foreground">Analysis Configuration</h2>
          </div>
          <div className="space-y-4">
            <div>
              <Label className="text-foreground mb-2 block">Rule Set</Label>
              <Select value={selectedRuleSet} onValueChange={setSelectedRuleSet}>
                <SelectTrigger className="bg-input border-border">
                  <SelectValue placeholder="Select rule set" />
                </SelectTrigger>
                <SelectContent>
                  {ruleSets.map((ruleSet: { id: string; name: string; language: string }) => (
                    <SelectItem key={ruleSet.id} value={ruleSet.id}>
                      {ruleSet.name} ({ruleSet.language})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-foreground mb-3 block">Priority</Label>
              <div className="grid grid-cols-3 gap-2">
                {(["normal", "high", "critical"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPriority(p)}
                    className={`py-2 px-4 rounded-lg text-sm font-medium transition-all capitalize ${
                      priority === p
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-foreground hover:bg-secondary/80"
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <Label className="text-foreground mb-3 block">Focus Areas</Label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { key: "securityScan", label: "Security detection" },
                  { key: "codeSmell", label: "Code smells & patterns" },
                  { key: "mlSuggestions", label: "Bug & complexity" },
                  { key: "refactoring", label: "Refactoring suggestions" },
                ].map((opt) => (
                  <div key={opt.key} className="flex items-center gap-3">
                    <Checkbox
                      id={opt.key}
                      checked={options[opt.key as keyof typeof options]}
                      onCheckedChange={(checked) =>
                        setOptions({ ...options, [opt.key]: !!checked })
                      }
                      className="border-primary data-[state=checked]:bg-primary"
                    />
                    <label htmlFor={opt.key} className="text-sm text-foreground cursor-pointer">
                      {opt.label}
                    </label>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                All 12 ML models always run — clones, debt, docs, performance & readability
                included.
              </p>
            </div>
          </div>
        </div>

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
