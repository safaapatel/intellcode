import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  UserPlus,
  Trash2,
  KeyRound,
  ShieldCheck,
  UserX,
  UserCheck,
  X,
  Users,
  AlertTriangle,
  Brain,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Database,
  MessageSquare,
} from "lucide-react";
import { toast } from "sonner";
import {
  getUsers,
  saveUsers,
  generatePassword,
  getSession,
  type StoredUser,
} from "@/services/auth";
import { getHealth, getModels, getStats, clearCache, getFeedbackStats } from "@/services/api";

const ROLE_BADGE: Record<StoredUser["role"], string> = {
  admin: "bg-primary/20 text-primary border border-primary/30",
  reviewer: "bg-accent/20 text-accent border border-accent/30",
  developer: "bg-secondary text-muted-foreground border border-border",
};

type ModalState =
  | { type: "add" }
  | { type: "delete"; user: StoredUser }
  | { type: "resetpw"; user: StoredUser; newPassword: string }
  | null;

const Admin = () => {
  const navigate = useNavigate();
  const session = getSession();
  const isAdmin = session?.role === "admin";

  const [users, setUsers] = useState<StoredUser[]>(() => getUsers());
  const [modal, setModal] = useState<ModalState>(null);
  const [search, setSearch] = useState("");

  // ── Model health state ──
  type ModelHealth = { models: Record<string, string>; cache: Record<string, number>; errors: Record<string, string> } | null;
  type ModelList = { models: Array<{ id: string; name: string; architecture: string; status: string; loaded?: boolean; [k: string]: unknown }> } | null;
  type BackendStats = { total_analyses: number; avg_score: number | null; min_score: number | null; max_score: number | null; total_security_findings: number; models_ready: number } | null;
  type FeedbackStats = { total: number; positive: number; negative: number; positive_rate: number } | null;
  const [health, setHealth] = useState<ModelHealth>(null);
  const [modelList, setModelList] = useState<ModelList>(null);
  const [backendStats, setBackendStats] = useState<BackendStats>(null);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  const [healthError, setHealthError] = useState(false);
  const [clearingCache, setClearingCache] = useState(false);

  const fetchModelStatus = async () => {
    setHealthLoading(true);
    setHealthError(false);
    try {
      const [h, m, s, fb] = await Promise.all([
        getHealth(),
        getModels(),
        getStats().catch(() => null),
        getFeedbackStats().catch(() => null),
      ]);
      setHealth(h as ModelHealth);
      setModelList(m as ModelList);
      setBackendStats(s as BackendStats);
      setFeedbackStats(fb as FeedbackStats);
    } catch {
      setHealthError(true);
    } finally {
      setHealthLoading(false);
    }
  };

  const handleClearCache = async () => {
    if (!window.confirm("Clear the backend analysis cache? All cached results will be removed and next analyses will re-run all models fresh.")) return;
    setClearingCache(true);
    try {
      await clearCache();
      toast.success("Backend cache cleared", { description: "Next analysis will re-run all models fresh." });
      fetchModelStatus();
    } catch (e) {
      toast.error("Could not clear cache", { description: e instanceof Error ? e.message : "Backend offline?" });
    } finally {
      setClearingCache(false);
    }
  };

  useEffect(() => { fetchModelStatus(); }, []);
  useEffect(() => { if (!isAdmin) navigate("/", { replace: true }); }, [isAdmin, navigate]);

  // Add user form state
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState<StoredUser["role"]>("developer");
  const [newPassword, setNewPassword] = useState("");
  const [addError, setAddError] = useState<string | null>(null);

  if (!isAdmin) return null;

  const persist = (updated: StoredUser[]) => {
    setUsers(updated);
    saveUsers(updated);
  };

  const handleAddUser = () => {
    setAddError(null);
    if (!newName.trim() || !newEmail.trim() || !newPassword.trim()) {
      setAddError("All fields are required.");
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(newEmail)) {
      setAddError("Please enter a valid email address.");
      return;
    }
    if (users.some((u) => u.email.toLowerCase() === newEmail.toLowerCase())) {
      setAddError("A user with this email already exists.");
      return;
    }
    const user: StoredUser = {
      id: Date.now().toString(),
      name: newName.trim(),
      email: newEmail.trim().toLowerCase(),
      password: newPassword.trim(),
      role: newRole,
      active: true,
      createdAt: new Date().toISOString(),
    };
    persist([...users, user]);
    toast.success(`User ${user.name} added`, { description: `Role: ${user.role}` });
    setModal(null);
    setNewName("");
    setNewEmail("");
    setNewPassword("");
    setNewRole("developer");
  };

  const handleDeleteUser = (user: StoredUser) => {
    if (user.id === session.userId) {
      toast.error("You cannot delete your own account.");
      return;
    }
    persist(users.filter((u) => u.id !== user.id));
    toast.success(`${user.name} removed`);
    setModal(null);
  };

  const handleToggleActive = (user: StoredUser) => {
    if (user.id === session.userId) {
      toast.error("You cannot deactivate your own account.");
      return;
    }
    persist(users.map((u) => (u.id === user.id ? { ...u, active: !u.active } : u)));
    toast.success(user.active ? `${user.name} deactivated` : `${user.name} reactivated`);
  };

  const handleChangeRole = (user: StoredUser, role: StoredUser["role"]) => {
    persist(users.map((u) => (u.id === user.id ? { ...u, role } : u)));
    toast.success(`${user.name}'s role updated to ${role}`);
  };

  const handleResetPassword = (user: StoredUser) => {
    const pw = generatePassword();
    persist(users.map((u) => (u.id === user.id ? { ...u, password: pw } : u)));
    setModal({ type: "resetpw", user, newPassword: pw });
  };

  const filtered = users.filter(
    (u) =>
      u.name.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase())
  );

  const stats = {
    total: users.length,
    active: users.filter((u) => u.active).length,
    admins: users.filter((u) => u.role === "admin").length,
    reviewers: users.filter((u) => u.role === "reviewer").length,
    developers: users.filter((u) => u.role === "developer").length,
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Breadcrumb */}
        <div className="text-sm text-muted-foreground mb-4">
          <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>
            Dashboard
          </span>
          <span className="mx-2">›</span>
          <span className="text-foreground">Admin</span>
        </div>

        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Admin Panel</h1>
            <p className="text-muted-foreground text-sm mt-1">Model status, user management, and system health</p>
          </div>
          <Button className="bg-gradient-primary gap-2" onClick={() => setModal({ type: "add" })}>
            <UserPlus className="w-4 h-4" />
            Add User
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
          {[
            { label: "Total Users", value: stats.total },
            { label: "Active", value: stats.active },
            { label: "Admins", value: stats.admins },
            { label: "Reviewers", value: stats.reviewers },
            { label: "Developers", value: stats.developers },
          ].map((s) => (
            <div key={s.label} className="bg-card border border-border rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-foreground">{s.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Search */}
        <div className="mb-4">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or email..."
            className="bg-input border-border max-w-xs"
          />
        </div>

        {/* Backend stats row */}
        {backendStats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            {[
              { label: "Cached Analyses", value: backendStats.total_analyses },
              { label: "Avg Score (cache)", value: backendStats.avg_score != null ? `${backendStats.avg_score}/100` : "—" },
              { label: "Security Findings", value: backendStats.total_security_findings },
              { label: "Models Ready", value: `${backendStats.models_ready}/12` },
            ].map((s) => (
              <div key={s.label} className="bg-card border border-border rounded-xl p-4 text-center">
                <p className="text-xl font-bold text-foreground">{s.value}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{s.label}</p>
              </div>
            ))}
          </div>
        )}

        {/* Feedback stats row */}
        {feedbackStats && feedbackStats.total > 0 && (
          <div className="bg-card border border-border rounded-xl p-5 mb-6">
            <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-primary" />
              Analysis Feedback
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Total Ratings", value: feedbackStats.total },
                { label: "Positive", value: `${feedbackStats.positive} 👍` },
                { label: "Satisfaction", value: `${Math.round(feedbackStats.positive_rate * 100)}%` },
              ].map((s) => (
                <div key={s.label} className="bg-secondary/30 rounded-lg p-3 text-center">
                  <p className="text-lg font-bold text-foreground">{s.value}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{s.label}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ML Model Status */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">ML Model Status</h2>
              {health && (
                <span className="text-xs bg-secondary px-2 py-0.5 rounded-full text-muted-foreground">
                  Cache: {(health.cache?.full_analysis_entries ?? 0)} full · {(health.cache?.specialist_entries ?? 0)} specialist
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleClearCache}
                disabled={clearingCache || healthLoading}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-destructive transition-colors"
              >
                <Database className={`w-3.5 h-3.5 ${clearingCache ? "opacity-50" : ""}`} />
                Clear Cache
              </button>
              <span className="text-border">|</span>
              <button
                onClick={fetchModelStatus}
                disabled={healthLoading}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${healthLoading ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>
          </div>

          {!health && !healthLoading && (
            <div className="bg-card border border-destructive/30 rounded-xl p-6 flex items-center gap-3 text-muted-foreground">
              <XCircle className="w-5 h-5 text-destructive shrink-0" />
              <p className="text-sm">
                {healthError ? "Backend unreachable." : "Model status not loaded."}{" "}
                Start the FastAPI server on port 8000, then{" "}
                <span className="text-primary cursor-pointer underline" onClick={fetchModelStatus}>
                  refresh
                </span>.
              </p>
            </div>
          )}

          {healthLoading && (
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-secondary/30 text-left">
                      <th className="py-3 px-4 text-muted-foreground font-medium">Model</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Architecture</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Loaded</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Registry Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: 6 }).map((_, i) => (
                      <tr key={i} className="border-b border-border/50">
                        {Array.from({ length: 4 }).map((__, j) => (
                          <td key={j} className="py-3 px-4">
                            <div className="h-4 bg-secondary/60 rounded animate-pulse" style={{ width: j === 0 ? "140px" : j === 1 ? "100px" : "60px" }} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {health && modelList && !healthLoading && (
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-secondary/30 text-left">
                      <th className="py-3 px-4 text-muted-foreground font-medium">Model</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Architecture</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Loaded</th>
                      <th className="py-3 px-4 text-muted-foreground font-medium">Registry Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelList.models.map((m) => {
                      const registryKey = m.id.replace(/-/g, "_");
                      const registryStatus = health.models[registryKey] ?? health.models[m.id] ?? "unknown";
                      const isReady = registryStatus === "ready";
                      const isLoaded = m.loaded !== false;
                      return (
                        <tr key={m.id} className="border-b border-border/50 hover:bg-secondary/10 transition-colors">
                          <td className="py-3 px-4">
                            <p className="font-medium text-foreground">{m.name}</p>
                            <p className="text-xs text-muted-foreground capitalize">{m.status?.replace(/_/g, " ")}</p>
                          </td>
                          <td className="py-3 px-4 text-xs text-muted-foreground font-mono leading-relaxed max-w-xs">
                            {m.architecture}
                          </td>
                          <td className="py-3 px-4">
                            {isLoaded ? (
                              <div className="flex items-center gap-1.5 text-green-400">
                                <Database className="w-3.5 h-3.5" />
                                <span className="text-xs">Loaded</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1.5 text-muted-foreground">
                                <Database className="w-3.5 h-3.5" />
                                <span className="text-xs">—</span>
                              </div>
                            )}
                          </td>
                          <td className="py-3 px-4">
                            {isReady ? (
                              <div className="flex items-center gap-1.5">
                                <CheckCircle2 className="w-4 h-4 text-green-400" />
                                <span className="text-xs text-green-400">Ready</span>
                              </div>
                            ) : registryStatus === "unavailable" ? (
                              <div className="flex items-center gap-1.5">
                                <XCircle className="w-4 h-4 text-destructive" />
                                <span className="text-xs text-destructive">Unavailable</span>
                              </div>
                            ) : (
                              <span className="text-xs text-muted-foreground capitalize">{registryStatus}</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {/* OOD Detectors */}
              {(health.models.ood_detector_security || health.models.ood_detector_bug) && (
                <div className="border-t border-border px-4 py-3 bg-secondary/10">
                  <p className="text-xs font-semibold text-muted-foreground mb-2">Out-of-Distribution Detectors</p>
                  <div className="flex gap-4">
                    {[
                      { key: "ood_detector_security", label: "Security OOD" },
                      { key: "ood_detector_bug", label: "Bug OOD" },
                    ].map(({ key, label }) => {
                      const status = health.models[key];
                      return (
                        <div key={key} className="flex items-center gap-1.5">
                          {status === "ready" ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                          ) : (
                            <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />
                          )}
                          <span className={`text-xs ${status === "ready" ? "text-green-400" : "text-yellow-400"}`}>
                            {label}: {status === "ready" ? "fitted" : "no checkpoint"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-1">
                    OOD detectors flag predictions on inputs far from the training distribution.
                    Train them by running <code className="font-mono">training/train_security.py</code> and{" "}
                    <code className="font-mono">training/train_bugs.py</code>.
                  </p>
                </div>
              )}

              {Object.keys(health.errors ?? {}).length > 0 && (
                <div className="border-t border-border px-4 py-3 bg-destructive/5">
                  <p className="text-xs font-semibold text-destructive mb-1">Load Errors</p>
                  {Object.entries(health.errors).map(([k, v]) => (
                    <p key={k} className="text-xs text-muted-foreground font-mono">
                      <span className="text-destructive">{k}:</span> {v}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Users heading */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Users</h2>
          </div>
        </div>

        {/* Table */}
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-secondary/30 text-left">
                  <th className="py-3 px-4 text-muted-foreground font-medium">Name</th>
                  <th className="py-3 px-4 text-muted-foreground font-medium">Email</th>
                  <th className="py-3 px-4 text-muted-foreground font-medium">Role</th>
                  <th className="py-3 px-4 text-muted-foreground font-medium">Status</th>
                  <th className="py-3 px-4 text-muted-foreground font-medium">Created</th>
                  <th className="py-3 px-4 text-muted-foreground font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-10 text-center text-muted-foreground">
                      <Users className="w-8 h-8 mx-auto mb-2 opacity-40" />
                      No users found
                    </td>
                  </tr>
                ) : (
                  filtered.map((user) => (
                    <tr
                      key={user.id}
                      className={`border-b border-border/50 hover:bg-secondary/10 transition-colors ${!user.active ? "opacity-50" : ""}`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center text-xs font-bold text-background">
                            {user.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()}
                          </div>
                          <div>
                            <p className="font-medium text-foreground">{user.name}</p>
                            {user.id === session.userId && (
                              <p className="text-xs text-primary">You</p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-muted-foreground font-mono text-xs">{user.email}</td>
                      <td className="py-3 px-4">
                        <Select
                          value={user.role}
                          onValueChange={(v) => handleChangeRole(user, v as StoredUser["role"])}
                          disabled={user.id === session.userId}
                        >
                          <SelectTrigger className="h-7 w-28 text-xs bg-transparent border-transparent hover:border-border">
                            <SelectValue>
                              <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${ROLE_BADGE[user.role]}`}>
                                {user.role}
                              </span>
                            </SelectValue>
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="admin">admin</SelectItem>
                            <SelectItem value="reviewer">reviewer</SelectItem>
                            <SelectItem value="developer">developer</SelectItem>
                          </SelectContent>
                        </Select>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                            user.active
                              ? "bg-green-500/20 text-green-400"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {user.active ? "Active" : "Inactive"}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-muted-foreground text-xs">
                        {new Date(user.createdAt).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-1">
                          <button
                            title={user.active ? "Deactivate" : "Reactivate"}
                            onClick={() => handleToggleActive(user)}
                            className="p-1.5 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                            disabled={user.id === session.userId}
                          >
                            {user.active ? <UserX className="w-4 h-4" /> : <UserCheck className="w-4 h-4" />}
                          </button>
                          <button
                            title="Reset password"
                            onClick={() => handleResetPassword(user)}
                            className="p-1.5 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <KeyRound className="w-4 h-4" />
                          </button>
                          <button
                            title="Delete user"
                            onClick={() => setModal({ type: "delete", user })}
                            className="p-1.5 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                            disabled={user.id === session.userId}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Add User Modal */}
      {modal?.type === "add" && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setModal(null)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-md w-full space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-foreground">Add New User</h3>
              <button onClick={() => setModal(null)} className="text-muted-foreground hover:text-foreground">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <Label className="text-foreground mb-1.5 block">Full Name</Label>
                <Input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Jane Smith"
                  className="bg-input border-border"
                />
              </div>
              <div>
                <Label className="text-foreground mb-1.5 block">Email</Label>
                <Input
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="jane@example.com"
                  className="bg-input border-border"
                />
              </div>
              <div>
                <Label className="text-foreground mb-1.5 block">Role</Label>
                <Select value={newRole} onValueChange={(v) => setNewRole(v as StoredUser["role"])}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="developer">Developer</SelectItem>
                    <SelectItem value="reviewer">Reviewer</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <Label className="text-foreground">Password</Label>
                  <button
                    type="button"
                    className="text-xs text-primary hover:underline"
                    onClick={() => setNewPassword(generatePassword())}
                  >
                    Generate
                  </button>
                </div>
                <Input
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Temporary password"
                  className="bg-input border-border font-mono"
                />
              </div>
            </div>

            {addError && <p className="text-sm text-destructive">{addError}</p>}

            <div className="flex gap-3 justify-end">
              <Button variant="outline" size="sm" onClick={() => setModal(null)}>
                Cancel
              </Button>
              <Button size="sm" className="bg-gradient-primary" onClick={handleAddUser}>
                Add User
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {modal?.type === "delete" && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setModal(null)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-sm w-full space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3">
              <div className="p-2 bg-destructive/10 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-destructive" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Delete User</h3>
                <p className="text-sm text-muted-foreground">This action cannot be undone.</p>
              </div>
            </div>
            <p className="text-sm text-foreground">
              Are you sure you want to permanently delete{" "}
              <span className="font-semibold">{modal.user.name}</span>?
            </p>
            <div className="flex gap-3 justify-end">
              <Button variant="outline" size="sm" onClick={() => setModal(null)}>
                Cancel
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => handleDeleteUser(modal.user)}
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Reset Password Result Modal */}
      {modal?.type === "resetpw" && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setModal(null)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-sm w-full space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-foreground">Password Reset</h3>
              <button onClick={() => setModal(null)} className="text-muted-foreground hover:text-foreground">
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-muted-foreground">
              New temporary password for <span className="text-foreground font-medium">{modal.user.name}</span>:
            </p>
            <div className="bg-secondary rounded-lg px-4 py-3 font-mono text-sm text-foreground break-all">
              {modal.newPassword}
            </div>
            <p className="text-xs text-muted-foreground">
              Share this securely. The user should change it after first login.
            </p>
            <div className="flex gap-3 justify-end">
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  navigator.clipboard.writeText(modal.newPassword);
                  toast.success("Password copied");
                }}
              >
                Copy
              </Button>
              <Button size="sm" className="bg-gradient-primary" onClick={() => setModal(null)}>
                Done
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Admin;
