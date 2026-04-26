import { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Github,
  Search,
  Plus,
  GitBranch,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ExternalLink,
  Settings,
  Trash2,
  RefreshCw,
  Star,
  ScanLine,
} from "lucide-react";
import { toast } from "sonner";
import { analyzeBatch } from "@/services/api";
import {
  isGitHubConnected,
  setGitHubToken,
  clearGitHubToken,
  getGitHubUser,
  listUserRepos,
  getRepoInfo,
  getRepoTree,
  getRawFile,
  type GitHubRepo,
  type GitHubUser,
} from "@/services/github";

const REPOS_KEY = "intellcode_repositories";

/** Extract Python source from Jupyter notebook JSON — discards outputs/images */
function extractNotebookCode(raw: string): string {
  try {
    const nb = JSON.parse(raw);
    return (nb.cells as Array<{ cell_type: string; source: string | string[] }>)
      .filter((c) => c.cell_type === "code")
      .map((c) => (Array.isArray(c.source) ? c.source.join("") : c.source))
      .filter((src) => src.trim().length > 0)
      .join("\n\n# ── next cell ──\n\n");
  } catch {
    return raw; // not valid JSON — send as-is
  }
}

const LANG_COLORS: Record<string, string> = {
  Python: "#3572A5",
  TypeScript: "#3178c6",
  JavaScript: "#f1e05a",
  Go: "#00ADD8",
  Java: "#b07219",
  Rust: "#dea584",
  "C++": "#f34b7d",
  C: "#555555",
};

// Files/dirs to skip when scanning
const SCAN_SKIP = [
  "venv/", "__pycache__/", "node_modules/", ".git/",
  "migrations/", "test_", "_test.py", "conftest.py",
  "setup.py", "checkpoints/", "data/", ".egg-info/",
  "training/", "generate_", "fetch_", "show_metrics",
];

type RepoStatus = "active" | "pending" | "error";

interface Repository {
  id: string;
  name: string;
  fullName: string;
  description: string;
  language: string;
  languageColor: string;
  stars: number;
  defaultBranch: string;
  lastReviewed: string;
  status: RepoStatus;
  totalReviews: number;
  openIssues: number;
  avgScore: number;
  private: boolean;
  githubUrl?: string;
}


function loadRepos(): Repository[] {
  try {
    const raw = localStorage.getItem(REPOS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveRepos(repos: Repository[]) {
  localStorage.setItem(REPOS_KEY, JSON.stringify(repos));
}

const languageOptions = ["All Languages", "Python", "TypeScript", "JavaScript", "Go", "Java"];
const statusOptions = ["All Status", "Active", "Pending", "Error"];

const statusConfig: Record<RepoStatus, { label: string; color: string; icon: typeof CheckCircle2 }> = {
  active:  { label: "Active",  color: "bg-green-500/20 text-green-400 border-green-500/30",   icon: CheckCircle2 },
  pending: { label: "Pending", color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", icon: Loader2 },
  error:   { label: "Error",   color: "bg-red-500/20 text-red-400 border-red-500/30",           icon: AlertCircle },
};

const Repositories = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const fromSubmit = (location.state as { from?: string } | null)?.from === "submit";

  const [repos, setRepos] = useState<Repository[]>([]);

  useEffect(() => { setRepos(loadRepos()); }, []);

  const [search, setSearch] = useState("");
  const [langFilter, setLangFilter] = useState("All Languages");
  const [statusFilter, setStatusFilter] = useState("All Status");
  const [connectOpen, setConnectOpen] = useState(false);
  const [newRepoUrl, setNewRepoUrl] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [scanning, setScanning] = useState<Record<string, boolean>>({});
  const [scanProgress, setScanProgress] = useState<Record<string, { done: number; total: number }>>({});

  // GitHub OAuth state
  const [ghConnected, setGhConnected] = useState(() => isGitHubConnected());
  const [ghUser, setGhUser] = useState<GitHubUser | null>(null);
  const [patOpen, setPatOpen] = useState(false);
  const [patValue, setPatValue] = useState("");
  const [patLoading, setPatLoading] = useState(false);
  const [ghRepos, setGhRepos] = useState<GitHubRepo[]>([]);
  const [importOpen, setImportOpen] = useState(false);
  const [loadingGhRepos, setLoadingGhRepos] = useState(false);
  const [importSearch, setImportSearch] = useState("");

  useEffect(() => {
    if (ghConnected) {
      getGitHubUser().then(setGhUser).catch(() => {
        setGhConnected(false);
      });
    }
  }, [ghConnected]);

  const handleConnectPAT = async () => {
    if (!patValue.trim()) { toast.error("Paste your GitHub token first"); return; }
    setPatLoading(true);
    try {
      setGitHubToken(patValue.trim());
      const user = await getGitHubUser();
      setGhUser(user);
      setGhConnected(true);
      setPatOpen(false);
      setPatValue("");
      toast.success(`Connected as @${user.login}`);
    } catch {
      clearGitHubToken();
      toast.error("Invalid token or no internet — check the token and try again");
    } finally {
      setPatLoading(false);
    }
  };

  const handleOpenImport = async () => {
    setImportOpen(true);
    if (ghRepos.length > 0) return;
    setLoadingGhRepos(true);
    try {
      const [p1, p2, p3] = await Promise.all([
        listUserRepos(1),
        listUserRepos(2),
        listUserRepos(3),
      ]);
      setGhRepos([...p1, ...p2, ...p3].filter((r) => !r.name.startsWith(".")));
    } catch (e: unknown) {
      toast.error((e as Error).message || "Failed to fetch repositories");
    } finally {
      setLoadingGhRepos(false);
    }
  };

  const handleImportRepo = (ghRepo: GitHubRepo) => {
    if (repos.some((r) => r.fullName.toLowerCase() === ghRepo.full_name.toLowerCase())) {
      toast.warning("Already connected");
      return;
    }
    const newRepo: Repository = {
      id: String(Date.now()),
      name: ghRepo.name,
      fullName: ghRepo.full_name,
      description: ghRepo.description || "No description",
      language: ghRepo.language || "Python",
      languageColor: LANG_COLORS[ghRepo.language || ""] || "#8b949e",
      stars: ghRepo.stargazers_count,
      defaultBranch: ghRepo.default_branch,
      lastReviewed: "Not yet scanned",
      status: "pending",
      totalReviews: 0,
      openIssues: 0,
      avgScore: 0,
      private: ghRepo.private,
      githubUrl: ghRepo.html_url,
    };
    setRepos((prev) => { const updated = [...prev, newRepo]; saveRepos(updated); return updated; });
    toast.success(`Connected ${ghRepo.full_name}`);
  };

  const filtered = repos.filter((r) => {
    const matchesSearch =
      r.fullName.toLowerCase().includes(search.toLowerCase()) ||
      r.description.toLowerCase().includes(search.toLowerCase());
    const matchesLang = langFilter === "All Languages" || r.language === langFilter;
    const matchesStatus =
      statusFilter === "All Status" || r.status === statusFilter.toLowerCase();
    return matchesSearch && matchesLang && matchesStatus;
  });

  // Connect: fetch real GitHub metadata
  const handleConnect = async () => {
    if (!newRepoUrl.trim()) { toast.error("Please enter a repository URL"); return; }
    setConnecting(true);
    try {
      const url = newRepoUrl.trim();
      const match = url.match(/github\.com\/([^/]+)\/([^/?#]+)/);
      if (!match) throw new Error("Invalid GitHub URL — expected https://github.com/owner/repo");
      const [, owner, repo] = match;

      const data = await getRepoInfo(`${owner}/${repo}`);

      // Check for duplicate
      if (repos.some((r) => r.fullName.toLowerCase() === data.full_name.toLowerCase())) {
        toast.warning("Repository already connected");
        return;
      }

      const newRepo: Repository = {
        id: String(Date.now()),
        name: data.name,
        fullName: data.full_name,
        description: data.description || "No description",
        language: data.language || "Python",
        languageColor: LANG_COLORS[data.language] || "#8b949e",
        stars: data.stargazers_count,
        defaultBranch: data.default_branch,
        lastReviewed: "Not yet scanned",
        status: "pending",
        totalReviews: 0,
        openIssues: 0,
        avgScore: 0,
        private: data.private,
        githubUrl: url,
      };
      setRepos((prev) => { const updated = [...prev, newRepo]; saveRepos(updated); return updated; });
      setConnectOpen(false);
      setNewRepoUrl("");
      toast.success(`Connected ${data.full_name} — click Scan to analyze`);
    } catch (e: unknown) {
      toast.error((e as Error).message || "Failed to connect repository");
    } finally {
      setConnecting(false);
    }
  };

  // Scan: fetch .py files from GitHub and run our ML analysis
  const scanRepo = async (repo: Repository) => {
    if (scanning[repo.id]) return;

    const fullName = repo.fullName;
    const branch = repo.defaultBranch || "main";

    setScanning((prev) => ({ ...prev, [repo.id]: true }));
    setScanProgress((prev) => ({ ...prev, [repo.id]: { done: 0, total: 0 } }));

    try {
      // 1. Fetch file tree (authenticated if token is available → 5000 req/hr vs 60)
      toast.info(`Fetching file tree for ${fullName}…`);
      let treeData: { tree?: Array<{ type: string; path: string; size?: number }> };
      try {
        treeData = await getRepoTree(fullName, branch);
      } catch (e: unknown) {
        const msg = (e as Error).message ?? "";
        throw new Error(
          msg.includes("409") || msg.includes("empty")
            ? "Repository is empty"
            : `Could not fetch file tree: ${msg}`
        );
      }

      // 2. Filter analyzable source files (all supported languages + Jupyter notebooks)
      const EXT_RE = /\.(py|js|ts|jsx|tsx|java|go|rs|cpp|cc|c|cs|rb|php|kt|swift|ipynb)$/;
      const EXT_LANG: Record<string, string> = {
        py: "python", js: "javascript", ts: "typescript",
        jsx: "javascript", tsx: "typescript", java: "java",
        go: "go", rs: "rust", cpp: "cpp", cc: "cpp",
        c: "c", cs: "csharp", rb: "ruby", php: "php",
        kt: "kotlin", swift: "swift", ipynb: "python",
      };
      const srcFiles: Array<{ path: string; size: number; lang: string }> = (treeData.tree || [])
        .filter(
          (f: { type: string; path: string; size?: number }) =>
            f.type === "blob" &&
            EXT_RE.test(f.path) &&
            (f.size ?? 0) < 500_000 &&
            !SCAN_SKIP.some((skip) => f.path.includes(skip))
        )
        .map((f: { path: string; size?: number }) => {
          const ext = f.path.split(".").pop()?.toLowerCase() ?? "";
          return { path: f.path, size: f.size ?? 0, lang: EXT_LANG[ext] ?? "python" };
        });

      if (srcFiles.length === 0) {
        toast.warning("No analyzable source files found (all skipped or too large)");
        setRepos((prev) => {
          const updated = prev.map((r) =>
            r.id === repo.id ? { ...r, status: "error" as RepoStatus } : r
          );
          saveRepos(updated);
          return updated;
        });
        return;
      }

      setScanProgress((prev) => ({ ...prev, [repo.id]: { done: 0, total: srcFiles.length } }));
      toast.info(`Scanning ${srcFiles.length} source file${srcFiles.length !== 1 ? "s" : ""}…`);

      // 3. Fetch all files concurrently (20 at a time)
      const FETCH_CONCURRENCY = 20;
      const fetched: Array<{ path: string; code: string; lang: string }> = [];
      for (let fi = 0; fi < srcFiles.length; fi += FETCH_CONCURRENCY) {
        const settled = await Promise.allSettled(
          srcFiles.slice(fi, fi + FETCH_CONCURRENCY).map(async (file) => {
            const raw = await getRawFile(fullName, branch, file.path);
            const code = file.path.endsWith(".ipynb") ? extractNotebookCode(raw) : raw;
            if (code.trim().length < 20) throw new Error("empty");
            return { path: file.path, code, lang: file.lang };
          })
        );
        for (const r of settled) {
          if (r.status === "fulfilled") fetched.push(r.value);
        }
      }

      // 4. Batch analyze (8 files per batch, 3 batches concurrent)
      const BATCH_SIZE = 8;
      const BATCH_CONCURRENCY = 3;
      const results: Array<{ overall_score?: number; security?: { findings?: unknown[]; summary?: { total?: number } } }> = [];
      let done = 0;
      setScanProgress((prev) => ({ ...prev, [repo.id]: { done: 0, total: fetched.length } }));
      const batches: typeof fetched[] = [];
      for (let bi = 0; bi < fetched.length; bi += BATCH_SIZE) batches.push(fetched.slice(bi, bi + BATCH_SIZE));
      for (let bi = 0; bi < batches.length; bi += BATCH_CONCURRENCY) {
        await Promise.all(
          batches.slice(bi, bi + BATCH_CONCURRENCY).map(async (batchFiles) => {
            try {
              const batchResult = await analyzeBatch(
                batchFiles.map((f) => ({ code: f.code, filename: f.path, language: f.lang }))
              );
              results.push(...batchResult.results);
            } catch { /* batch failed, count as skipped */ }
            done += batchFiles.length;
            setScanProgress((prev) => ({ ...prev, [repo.id]: { done, total: fetched.length } }));
          })
        );
      }

      // 4. Aggregate
      if (results.length === 0) {
        toast.error("No files could be analyzed — is the backend running?");
        setRepos((prev) => {
          const updated = prev.map((r) =>
            r.id === repo.id ? { ...r, status: "error" as RepoStatus } : r
          );
          saveRepos(updated);
          return updated;
        });
        return;
      }
      const avgScore = Math.round(
        results.reduce((a, r) => a + (r.overall_score ?? 0), 0) / results.length
      );
      const openIssues = results.reduce(
        (a, r) => a + (r.security?.findings?.length ?? r.security?.summary?.total ?? 0),
        0
      );

      setRepos((prev) => {
        const updated = prev.map((r) =>
          r.id === repo.id
            ? {
                ...r,
                status: "active" as RepoStatus,
                totalReviews: results.length,
                openIssues,
                avgScore,
                lastReviewed: "Just now",
              }
            : r
        );
        saveRepos(updated);
        return updated;
      });

      toast.success(
        `Scanned ${results.length} of ${srcFiles.length} files — avg score ${avgScore}/100, ${openIssues} security issue${openIssues !== 1 ? "s" : ""}`,
        skipped > 0 ? { description: `${skipped} file${skipped !== 1 ? "s" : ""} skipped (empty or fetch failed)` } : undefined
      );
    } catch (e: unknown) {
      toast.error((e as Error).message || "Scan failed");
      setRepos((prev) => {
        const updated = prev.map((r) =>
          r.id === repo.id ? { ...r, status: "error" as RepoStatus } : r
        );
        saveRepos(updated);
        return updated;
      });
    } finally {
      setScanning((prev) => ({ ...prev, [repo.id]: false }));
      setScanProgress((prev) => { const p = { ...prev }; delete p[repo.id]; return p; });
    }
  };

  const handleDelete = (id: string) => {
    setRepos((prev) => { const updated = prev.filter((r) => r.id !== id); saveRepos(updated); return updated; });
    setDeleteId(null);
    toast.success("Repository disconnected");
  };

  const totalReviews = repos.reduce((a, r) => a + r.totalReviews, 0);
  const scoredRepos = repos.filter((r) => r.avgScore > 0);
  const avgScore = scoredRepos.length
    ? Math.round(scoredRepos.reduce((a, r) => a + r.avgScore, 0) / scoredRepos.length)
    : 0;

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Back-to-Submit banner */}
        {fromSubmit && (
          <div className="flex items-center justify-between bg-primary/10 border border-primary/20 rounded-xl px-5 py-3 mb-6">
            <p className="text-sm text-foreground">Connect or select a repository, then go back to Submit to scan its files.</p>
            <Button size="sm" className="gap-1.5 bg-gradient-primary shrink-0" onClick={() => navigate("/submit")}>
              ← Back to Submit
            </Button>
          </div>
        )}
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Repositories</h1>
            <p className="text-muted-foreground mt-1">
              {ghConnected && ghUser
                ? `Connected as @${ghUser.login} · ${ghUser.public_repos + (ghUser.private_repos ?? 0)} repos available`
                : "Connect GitHub repositories and scan with ML models"}
            </p>
          </div>
          <div className="flex gap-2">
            {ghConnected ? (
              <Button variant="outline" className="gap-2" onClick={handleOpenImport}>
                <Github className="w-4 h-4" />
                Import from GitHub
              </Button>
            ) : (
              <Button variant="outline" className="gap-2" onClick={() => setPatOpen(true)}>
                <Github className="w-4 h-4" />
                Connect GitHub
              </Button>
            )}
            <Button className="bg-gradient-primary gap-2" onClick={() => setConnectOpen(true)}>
              <Plus className="w-4 h-4" />
              Add by URL
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Connected Repos", value: repos.length, sub: `${repos.filter((r) => r.status === "active").length} active` },
            { label: "Files Analyzed",  value: totalReviews, sub: "across all repos" },
            { label: "Avg Quality Score", value: avgScore > 0 ? `${avgScore}/100` : "—", sub: "combined score" },
            { label: "Security Issues", value: repos.reduce((a, r) => a + r.openIssues, 0), sub: "needs attention" },
          ].map(({ label, value, sub }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-4">
              <div className="text-2xl font-bold text-foreground mb-1">{value}</div>
              <div className="text-sm font-medium text-foreground">{label}</div>
              <div className="text-xs text-muted-foreground">{sub}</div>
            </div>
          ))}
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-6">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search repositories..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10 bg-card border-border"
            />
          </div>
          <Select value={langFilter} onValueChange={setLangFilter}>
            <SelectTrigger className="w-44 bg-card border-border"><SelectValue /></SelectTrigger>
            <SelectContent>
              {languageOptions.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36 bg-card border-border"><SelectValue /></SelectTrigger>
            <SelectContent>
              {statusOptions.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>

        {/* Repositories */}
        <div className="space-y-3">
          {filtered.length === 0 ? (
            <div className="text-center py-16 text-muted-foreground">
              <Github className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p className="font-medium">No repositories found</p>
              <p className="text-sm mt-1">Try adjusting your search or filters</p>
            </div>
          ) : (
            filtered.map((repo) => {
              const { label, color } = statusConfig[repo.status];
              const isScanning = scanning[repo.id] ?? false;
              const progress = scanProgress[repo.id];
              return (
                <div
                  key={repo.id}
                  className="bg-card border border-border rounded-xl p-5 hover:border-primary/30 transition-colors"
                >
                  <div className="flex items-start justify-between gap-4">
                    {/* Left */}
                    <div className="flex items-start gap-4 min-w-0 flex-1">
                      <div className="w-10 h-10 bg-secondary rounded-lg flex items-center justify-center flex-shrink-0">
                        <Github className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <span className="font-semibold text-foreground">{repo.fullName}</span>
                          {repo.private && (
                            <Badge className="bg-secondary text-muted-foreground border-border text-xs">Private</Badge>
                          )}
                          <Badge className={`${color} text-xs`}>{label}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-3 truncate">{repo.description}</p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                          <span className="flex items-center gap-1">
                            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: repo.languageColor }} />
                            {repo.language}
                          </span>
                          <span className="flex items-center gap-1"><Star className="w-3 h-3" />{repo.stars}</span>
                          <span className="flex items-center gap-1"><GitBranch className="w-3 h-3" />{repo.defaultBranch}</span>
                          <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{repo.lastReviewed}</span>
                        </div>

                        {/* Scan progress bar */}
                        {isScanning && progress && (
                          <div className="mt-3">
                            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                              <span className="flex items-center gap-1">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Scanning… {progress.done}/{progress.total} files
                              </span>
                              <span>{progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0}%</span>
                            </div>
                            <div className="w-full bg-secondary rounded-full h-1.5">
                              <div
                                className="bg-primary h-1.5 rounded-full transition-all duration-300"
                                style={{ width: progress.total > 0 ? `${(progress.done / progress.total) * 100}%` : "0%" }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Right: stats + actions */}
                    <div className="flex items-center gap-6 flex-shrink-0">
                      {repo.status === "active" && (
                        <div className="hidden md:flex gap-5 text-center">
                          <div>
                            <div className="text-lg font-bold text-foreground">{repo.totalReviews}</div>
                            <div className="text-xs text-muted-foreground">Files</div>
                          </div>
                          <div>
                            <div className={`text-lg font-bold ${repo.openIssues > 5 ? "text-orange-400" : "text-foreground"}`}>
                              {repo.openIssues}
                            </div>
                            <div className="text-xs text-muted-foreground">Issues</div>
                          </div>
                          <div>
                            <div className={`text-lg font-bold ${repo.avgScore >= 85 ? "text-green-400" : repo.avgScore >= 70 ? "text-yellow-400" : "text-red-400"}`}>
                              {repo.avgScore}
                            </div>
                            <div className="text-xs text-muted-foreground">Score</div>
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-1">
                        {/* Scan button */}
                        <Button
                          size="sm"
                          variant={repo.status === "pending" ? "default" : "ghost"}
                          className={repo.status === "pending" ? "gap-1.5 bg-gradient-primary text-xs px-3" : "gap-1.5 text-muted-foreground hover:text-foreground"}
                          onClick={() => scanRepo(repo)}
                          disabled={isScanning}
                          title="Scan repository"
                        >
                          {isScanning
                            ? <Loader2 className="w-4 h-4 animate-spin" />
                            : repo.status === "pending"
                              ? <><ScanLine className="w-4 h-4" /><span>Scan</span></>
                              : <RefreshCw className="w-4 h-4" />
                          }
                        </Button>
                        <Button size="sm" variant="ghost" className="gap-1.5 text-muted-foreground hover:text-foreground" asChild>
                          <Link to={`/reviews?repo=${encodeURIComponent(repo.fullName)}`} title="View reviews"><ExternalLink className="w-4 h-4" /></Link>
                        </Button>
                        <Button size="sm" variant="ghost" className="gap-1.5 text-muted-foreground hover:text-foreground" asChild title="Settings">
                          <Link to="/rules"><Settings className="w-4 h-4" /></Link>
                        </Button>
                        <Button
                          size="sm" variant="ghost"
                          className="gap-1.5 text-destructive hover:bg-destructive/10"
                          onClick={() => setDeleteId(repo.id)}
                          title="Disconnect"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </main>

      {/* GitHub PAT connect dialog */}
      <Dialog open={patOpen} onOpenChange={setPatOpen}>
        <DialogContent className="bg-card border-border max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Github className="w-5 h-5" /> Connect GitHub
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <p className="text-sm text-muted-foreground">
              Paste a GitHub Personal Access Token (PAT) with <code className="bg-secondary px-1 rounded text-xs">repo</code> scope.
              Create one at{" "}
              <a href="https://github.com/settings/tokens/new?scopes=repo&description=IntelliCode" target="_blank" rel="noreferrer" className="text-primary underline">
                github.com/settings/tokens
              </a>.
            </p>
            <Input
              type="password"
              placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
              value={patValue}
              onChange={(e) => setPatValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleConnectPAT(); }}
              className="font-mono bg-input border-border"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPatOpen(false)}>Cancel</Button>
            <Button className="bg-gradient-primary gap-2" onClick={handleConnectPAT} disabled={patLoading}>
              {patLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Github className="w-4 h-4" />}
              Connect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Connect Repository Dialog */}
      <Dialog open={connectOpen} onOpenChange={setConnectOpen}>
        <DialogContent className="bg-card border-border max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Github className="w-5 h-5" />
              Connect Repository
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div>
              <p className="text-xs font-semibold text-muted-foreground mb-2">ENTER GITHUB REPOSITORY URL</p>
              <Input
                placeholder="https://github.com/owner/repo"
                value={newRepoUrl}
                onChange={(e) => setNewRepoUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleConnect()}
                className="bg-input"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Public repositories only · no token required
              </p>
            </div>

            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">QUICK ADD</p>
              <div className="space-y-2 max-h-36 overflow-y-auto">
                {[
                  "https://github.com/safaapatel/intellcode",
                  "https://github.com/pallets/flask",
                  "https://github.com/psf/requests",
                ].map((r) => (
                  <button
                    key={r}
                    className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-secondary text-sm text-foreground"
                    onClick={() => setNewRepoUrl(r)}
                  >
                    <Github className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                    <span className="truncate">{r.replace("https://github.com/", "")}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-secondary/20 border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground">
                After connecting, click <strong className="text-foreground">Scan</strong> to fetch Python files and run all ML models.
                Up to 20 files are analyzed per scan.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setConnectOpen(false)}>Cancel</Button>
            <Button className="bg-gradient-primary gap-2" onClick={handleConnect} disabled={connecting}>
              {connecting && <Loader2 className="w-4 h-4 animate-spin" />}
              {connecting ? "Connecting…" : "Connect"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Disconnect Confirmation */}
      <Dialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <DialogContent className="bg-card border-border max-w-sm">
          <DialogHeader>
            <DialogTitle>Disconnect Repository</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground py-2">
            This will remove the repository from IntelliCode. Existing review history in the Reviews page will be preserved.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => deleteId && handleDelete(deleteId)}>Disconnect</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import from GitHub — list all authenticated user repos */}
      <Dialog open={importOpen} onOpenChange={setImportOpen}>
        <DialogContent className="bg-card border-border max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Github className="w-5 h-5" />
              {ghUser ? `@${ghUser.login}'s Repositories` : "Your Repositories"}
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                className="w-full pl-10 pr-4 py-2 text-sm bg-input border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="Search repositories…"
                value={importSearch}
                onChange={(e) => setImportSearch(e.target.value)}
              />
            </div>

            <div className="max-h-80 overflow-y-auto space-y-1 pr-1">
              {loadingGhRepos ? (
                <div className="flex items-center justify-center py-8 gap-2 text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading repositories…
                </div>
              ) : ghRepos.filter((r) =>
                  r.full_name.toLowerCase().includes(importSearch.toLowerCase()) ||
                  (r.description ?? "").toLowerCase().includes(importSearch.toLowerCase())
                ).length === 0 ? (
                <div className="text-center py-8 text-muted-foreground text-sm">No repositories found</div>
              ) : (
                ghRepos
                  .filter((r) =>
                    r.full_name.toLowerCase().includes(importSearch.toLowerCase()) ||
                    (r.description ?? "").toLowerCase().includes(importSearch.toLowerCase())
                  )
                  .map((r) => {
                    const alreadyAdded = repos.some((x) => x.fullName.toLowerCase() === r.full_name.toLowerCase());
                    return (
                      <div
                        key={r.id}
                        className="flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg hover:bg-secondary/50 transition-colors"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-foreground truncate">{r.full_name}</span>
                            {r.private && (
                              <Badge className="bg-secondary text-muted-foreground border-border text-xs flex-shrink-0">Private</Badge>
                            )}
                          </div>
                          {r.description && (
                            <p className="text-xs text-muted-foreground truncate mt-0.5">{r.description}</p>
                          )}
                          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                            {r.language && (
                              <span className="flex items-center gap-1">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: LANG_COLORS[r.language] || "#8b949e" }} />
                                {r.language}
                              </span>
                            )}
                            <span className="flex items-center gap-1"><Star className="w-3 h-3" />{r.stargazers_count}</span>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant={alreadyAdded ? "ghost" : "outline"}
                          className={alreadyAdded ? "text-muted-foreground" : "gap-1.5"}
                          disabled={alreadyAdded}
                          onClick={() => handleImportRepo(r)}
                        >
                          {alreadyAdded ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <Plus className="w-3.5 h-3.5" />}
                          {alreadyAdded ? "Added" : "Add"}
                        </Button>
                      </div>
                    );
                  })
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setImportOpen(false)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Repositories;
