import { useState } from "react";
import { Link } from "react-router-dom";
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
} from "lucide-react";
import { toast } from "sonner";

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
}

const mockRepos: Repository[] = [
  {
    id: "1",
    name: "code-review",
    fullName: "safaa-patel/code-review",
    description: "Main code review service with ML-powered analysis",
    language: "Python",
    languageColor: "#3572A5",
    stars: 12,
    defaultBranch: "main",
    lastReviewed: "2 hours ago",
    status: "active",
    totalReviews: 67,
    openIssues: 5,
    avgScore: 87,
    private: false,
  },
  {
    id: "2",
    name: "react-app",
    fullName: "frontend/react-app",
    description: "Frontend React application for the review dashboard",
    language: "TypeScript",
    languageColor: "#3178c6",
    stars: 8,
    defaultBranch: "main",
    lastReviewed: "1 day ago",
    status: "active",
    totalReviews: 43,
    openIssues: 3,
    avgScore: 91,
    private: false,
  },
  {
    id: "3",
    name: "api-service",
    fullName: "backend/api-service",
    description: "REST API backend service with FastAPI",
    language: "Python",
    languageColor: "#3572A5",
    stars: 5,
    defaultBranch: "develop",
    lastReviewed: "3 days ago",
    status: "active",
    totalReviews: 29,
    openIssues: 7,
    avgScore: 79,
    private: true,
  },
  {
    id: "4",
    name: "training",
    fullName: "ml-models/training",
    description: "ML model training scripts and experiments",
    language: "Python",
    languageColor: "#3572A5",
    stars: 3,
    defaultBranch: "main",
    lastReviewed: "Pending first review",
    status: "pending",
    totalReviews: 0,
    openIssues: 0,
    avgScore: 0,
    private: true,
  },
];

const languageOptions = ["All Languages", "Python", "TypeScript", "JavaScript", "Go", "Java"];
const statusOptions = ["All Status", "Active", "Pending", "Error"];

const statusConfig: Record<RepoStatus, { label: string; color: string; icon: typeof CheckCircle2 }> = {
  active: { label: "Active", color: "bg-green-500/20 text-green-400 border-green-500/30", icon: CheckCircle2 },
  pending: { label: "Pending", color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", icon: Loader2 },
  error: { label: "Error", color: "bg-red-500/20 text-red-400 border-red-500/30", icon: AlertCircle },
};

const Repositories = () => {
  const [repos, setRepos] = useState(mockRepos);
  const [search, setSearch] = useState("");
  const [langFilter, setLangFilter] = useState("All Languages");
  const [statusFilter, setStatusFilter] = useState("All Status");
  const [connectOpen, setConnectOpen] = useState(false);
  const [newRepoUrl, setNewRepoUrl] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const filtered = repos.filter((r) => {
    const matchesSearch =
      r.fullName.toLowerCase().includes(search.toLowerCase()) ||
      r.description.toLowerCase().includes(search.toLowerCase());
    const matchesLang = langFilter === "All Languages" || r.language === langFilter;
    const matchesStatus =
      statusFilter === "All Status" || r.status === statusFilter.toLowerCase();
    return matchesSearch && matchesLang && matchesStatus;
  });

  const handleConnect = () => {
    if (!newRepoUrl.trim()) {
      toast.error("Please enter a repository URL");
      return;
    }
    setConnecting(true);
    setTimeout(() => {
      setConnecting(false);
      setConnectOpen(false);
      setNewRepoUrl("");
      toast.success("Repository connected successfully");
    }, 1800);
  };

  const handleDelete = (id: string) => {
    setRepos((prev) => prev.filter((r) => r.id !== id));
    setDeleteId(null);
    toast.success("Repository disconnected");
  };

  const handleSync = (name: string) => {
    toast.info(`Syncing ${name}...`);
    setTimeout(() => toast.success(`${name} synced`), 1500);
  };

  const totalReviews = repos.reduce((a, r) => a + r.totalReviews, 0);
  const avgScore = Math.round(
    repos.filter((r) => r.avgScore > 0).reduce((a, r) => a + r.avgScore, 0) /
      repos.filter((r) => r.avgScore > 0).length
  );

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Repositories</h1>
            <p className="text-muted-foreground mt-1">
              Manage connected GitHub repositories and review configurations
            </p>
          </div>
          <Button className="bg-gradient-primary gap-2" onClick={() => setConnectOpen(true)}>
            <Plus className="w-4 h-4" />
            Connect Repository
          </Button>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Connected Repos", value: repos.length, sub: `${repos.filter((r) => r.status === "active").length} active` },
            { label: "Total Reviews", value: totalReviews, sub: "across all repos" },
            { label: "Avg Quality Score", value: `${avgScore}/100`, sub: "combined score" },
            { label: "Open Issues", value: repos.reduce((a, r) => a + r.openIssues, 0), sub: "needs attention" },
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
            <SelectTrigger className="w-44 bg-card border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languageOptions.map((l) => (
                <SelectItem key={l} value={l}>{l}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36 bg-card border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {statusOptions.map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Repositories Grid */}
        <div className="space-y-3">
          {filtered.length === 0 ? (
            <div className="text-center py-16 text-muted-foreground">
              <Github className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p className="font-medium">No repositories found</p>
              <p className="text-sm mt-1">Try adjusting your search or filters</p>
            </div>
          ) : (
            filtered.map((repo) => {
              const { label, color, icon: StatusIcon } = statusConfig[repo.status];
              return (
                <div
                  key={repo.id}
                  className="bg-card border border-border rounded-xl p-5 hover:border-primary/30 transition-colors"
                >
                  <div className="flex items-start justify-between gap-4">
                    {/* Left: info */}
                    <div className="flex items-start gap-4 min-w-0">
                      <div className="w-10 h-10 bg-secondary rounded-lg flex items-center justify-center flex-shrink-0">
                        <Github className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <span className="font-semibold text-foreground">{repo.fullName}</span>
                          {repo.private && (
                            <Badge className="bg-secondary text-muted-foreground border-border text-xs">Private</Badge>
                          )}
                          <Badge className={`${color} text-xs`}>
                            {label}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-3 truncate">{repo.description}</p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                          <span className="flex items-center gap-1">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: repo.languageColor }}
                            />
                            {repo.language}
                          </span>
                          <span className="flex items-center gap-1">
                            <Star className="w-3 h-3" />
                            {repo.stars}
                          </span>
                          <span className="flex items-center gap-1">
                            <GitBranch className="w-3 h-3" />
                            {repo.defaultBranch}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {repo.lastReviewed}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Right: stats + actions */}
                    <div className="flex items-center gap-6 flex-shrink-0">
                      {repo.status === "active" && (
                        <div className="hidden md:flex gap-5 text-center">
                          <div>
                            <div className="text-lg font-bold text-foreground">{repo.totalReviews}</div>
                            <div className="text-xs text-muted-foreground">Reviews</div>
                          </div>
                          <div>
                            <div className={`text-lg font-bold ${repo.openIssues > 5 ? "text-orange-400" : "text-foreground"}`}>
                              {repo.openIssues}
                            </div>
                            <div className="text-xs text-muted-foreground">Open Issues</div>
                          </div>
                          <div>
                            <div className={`text-lg font-bold ${repo.avgScore >= 85 ? "text-green-400" : repo.avgScore >= 70 ? "text-yellow-400" : "text-red-400"}`}>
                              {repo.avgScore}
                            </div>
                            <div className="text-xs text-muted-foreground">Avg Score</div>
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          className="gap-1.5 text-muted-foreground hover:text-foreground"
                          onClick={() => handleSync(repo.name)}
                          title="Sync"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="gap-1.5 text-muted-foreground hover:text-foreground"
                          asChild
                        >
                          <Link to="/submit" title="Submit for review">
                            <ExternalLink className="w-4 h-4" />
                          </Link>
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="gap-1.5 text-muted-foreground hover:text-foreground"
                          asChild
                          title="Settings"
                        >
                          <Link to="/rules">
                            <Settings className="w-4 h-4" />
                          </Link>
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
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
              <label className="text-sm font-medium text-foreground mb-1.5 block">
                GitHub Repository URL
              </label>
              <Input
                placeholder="https://github.com/owner/repo"
                value={newRepoUrl}
                onChange={(e) => setNewRepoUrl(e.target.value)}
                className="bg-input"
              />
            </div>

            <div className="bg-secondary/30 border border-border rounded-lg p-3 text-sm text-muted-foreground">
              Connecting a repository will install the IntelliCode Review GitHub App and trigger automatic reviews on every pull request.
            </div>

            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">OR CHOOSE FROM YOUR GITHUB REPOS</p>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {["safaa-patel/ml-experiments", "safaa-patel/fastapi-template", "safaa-patel/data-pipeline"].map((r) => (
                  <button
                    key={r}
                    className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-secondary text-sm text-foreground"
                    onClick={() => setNewRepoUrl(`https://github.com/${r}`)}
                  >
                    <Github className="w-4 h-4 text-muted-foreground" />
                    {r}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setConnectOpen(false)}>Cancel</Button>
            <Button className="bg-gradient-primary gap-2" onClick={handleConnect} disabled={connecting}>
              {connecting && <Loader2 className="w-4 h-4 animate-spin" />}
              {connecting ? "Connecting..." : "Connect"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Disconnect Confirmation Dialog */}
      <Dialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <DialogContent className="bg-card border-border max-w-sm">
          <DialogHeader>
            <DialogTitle>Disconnect Repository</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground py-2">
            This will remove the repository from IntelliCode Review and uninstall the GitHub App webhook. Existing review history will be preserved.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => deleteId && handleDelete(deleteId)}>
              Disconnect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Repositories;
