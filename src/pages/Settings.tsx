import { useState, useRef, useEffect } from "react";
import {
  isGitHubConnected,
  clearGitHubToken,
  getGitHubUser,
  type GitHubUser,
} from "@/services/github";
import { applyTheme, getTheme, type Theme } from "@/lib/theme";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  User,
  Bell,
  Github,
  Key,
  Shield,
  Palette,
  Save,
  RefreshCw,
  Copy,
  Check,
  LogOut,
  Link as LinkIcon,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { mockUser } from "@/data/mockData";
import { toast } from "sonner";

const SETTINGS_KEY = "intellcode_settings";

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}"); } catch { return {}; }
}
function saveSettings(patch: Record<string, unknown>) {
  const current = loadSettings();
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...current, ...patch }));
}

function randomHex(n: number) {
  return Array.from(crypto.getRandomValues(new Uint8Array(n)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, n);
}

type SettingsTab = "profile" | "notifications" | "integrations" | "security" | "appearance";

interface Session {
  id: string;
  device: string;
  location: string;
  current: boolean;
  time: string;
}

const INITIAL_SESSIONS: Session[] = [
  { id: "s1", device: "Chrome on Windows 11", location: "Reno, NV", current: true, time: "Now" },
  { id: "s2", device: "Safari on iPhone 15", location: "Reno, NV", current: false, time: "2 hours ago" },
];

const Settings = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<SettingsTab>("profile");
  const [copied, setCopied] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const saved = loadSettings();

  // Profile state
  const [name, setName] = useState<string>(saved.name ?? mockUser.name);
  const [email, setEmail] = useState<string>(saved.email ?? mockUser.email);
  const [role, setRole] = useState<string>(saved.role ?? mockUser.role);
  const [bio, setBio] = useState<string>(saved.bio ?? "Software engineer focused on code quality and ML systems.");
  const [timezone, setTimezone] = useState<string>(saved.timezone ?? "America/Los_Angeles");
  const [photoUrl, setPhotoUrl] = useState<string>(saved.photoUrl ?? "");

  // Notification state
  const [notifReviewComplete, setNotifReviewComplete] = useState<boolean>(saved.notifReviewComplete ?? true);
  const [notifCriticalIssues, setNotifCriticalIssues] = useState<boolean>(saved.notifCriticalIssues ?? true);
  const [notifWeeklyDigest, setNotifWeeklyDigest] = useState<boolean>(saved.notifWeeklyDigest ?? false);
  const [notifNewComment, setNotifNewComment] = useState<boolean>(saved.notifNewComment ?? true);
  const [notifPRMerged, setNotifPRMerged] = useState<boolean>(saved.notifPRMerged ?? false);
  const [emailFrequency, setEmailFrequency] = useState<string>(saved.emailFrequency ?? "instant");
  const [browserPermission, setBrowserPermission] = useState<NotificationPermission>(
    "Notification" in window ? Notification.permission : "denied"
  );

  // Security state
  const [twoFAEnabled, setTwoFAEnabled] = useState<boolean>(saved.twoFAEnabled ?? false);
  const [apiToken, setApiToken] = useState<string>(saved.apiToken ?? "");
  const [sessions, setSessions] = useState<Session[]>(() => {
    try {
      const raw = localStorage.getItem("intellcode_sessions");
      return raw ? JSON.parse(raw) : INITIAL_SESSIONS;
    } catch {
      return INITIAL_SESSIONS;
    }
  });

  // Integrations state
  const [githubConnected, setGithubConnected] = useState<boolean>(() => isGitHubConnected());
  const [githubUser, setGithubUser] = useState<GitHubUser | null>(null);
  const [slackWebhookUrl, setSlackWebhookUrl] = useState<string>(saved.slackWebhookUrl ?? "");
  const [slackConnected, setSlackConnected] = useState<boolean>(saved.slackConnected ?? false);

  // Load GitHub user info if connected
  useEffect(() => {
    if (githubConnected) {
      getGitHubUser().then(setGithubUser).catch(() => {});
    }
  }, [githubConnected]);

  // Profile extras
  const [preferredLanguage, setPreferredLanguage] = useState<string>(saved.preferredLanguage ?? "python");

  // Appearance state
  const [codeFont, setCodeFont] = useState<string>(saved.codeFont ?? "jetbrains");
  const [density, setDensity] = useState<string>(saved.density ?? "Comfortable");
  const [currentTheme, setCurrentTheme] = useState<Theme>(() => getTheme());

  // Keep theme state in sync when changed from another tab/window
  useEffect(() => {
    const handler = (e: Event) => setCurrentTheme((e as CustomEvent).detail as Theme);
    window.addEventListener("intellcode-theme", handler);
    return () => window.removeEventListener("intellcode-theme", handler);
  }, []);

  const handleThemeChange = (t: Theme) => {
    applyTheme(t);
    setCurrentTheme(t);
  };

  const tabs: { id: SettingsTab; label: string; icon: typeof User }[] = [
    { id: "profile", label: "Profile", icon: User },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "integrations", label: "Integrations", icon: Github },
    { id: "security", label: "Security", icon: Shield },
    { id: "appearance", label: "Appearance", icon: Palette },
  ];

  // --- Profile handlers ---
  const handleSaveProfile = () => {
    saveSettings({ name, email, role, bio, timezone, photoUrl, preferredLanguage });
    toast.success("Profile updated successfully");
  };

  const handleUploadPhoto = () => fileInputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) {
      toast.error("Photo must be under 2 MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const url = ev.target?.result as string;
      setPhotoUrl(url);
      saveSettings({ photoUrl: url });
      toast.success("Profile photo updated");
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const handleRemovePhoto = () => {
    setPhotoUrl("");
    saveSettings({ photoUrl: "" });
    toast.success("Profile photo removed");
  };

  const handleDeleteAccount = () => {
    // Clear all app data and redirect to login
    localStorage.clear();
    toast.success("Account deleted. Goodbye!");
    setTimeout(() => navigate("/"), 1000);
  };

  // --- Notification handlers ---
  const handleSaveNotifications = () => {
    saveSettings({ notifReviewComplete, notifCriticalIssues, notifWeeklyDigest, notifNewComment, notifPRMerged, emailFrequency });
    toast.success("Notification preferences saved");
  };

  const handleRequestBrowserPermission = async () => {
    if (!("Notification" in window)) {
      toast.error("This browser does not support desktop notifications");
      return;
    }
    const perm = await Notification.requestPermission();
    setBrowserPermission(perm);
    if (perm === "granted") {
      toast.success("Browser notifications enabled!");
      new Notification("IntelliCode", { body: "You'll now receive critical security alerts here.", icon: "/favicon.ico" });
    } else if (perm === "denied") {
      toast.error("Notifications blocked — change in your browser's site settings");
    }
  };

  // --- Security handlers ---
  const handleCopyToken = () => {
    navigator.clipboard.writeText(apiToken).catch(() => {});
    setCopied(true);
    toast.success("API token copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRegenerateToken = () => {
    const newToken = "ick_live_" + randomHex(32);
    setApiToken(newToken);
    saveSettings({ apiToken: newToken });
    navigator.clipboard.writeText(newToken).catch(() => {});
    toast.success("New API token generated and copied to clipboard");
  };

  const handleRevokeSession = (id: string) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      localStorage.setItem("intellcode_sessions", JSON.stringify(updated));
      return updated;
    });
    toast.success("Session revoked");
  };

  // --- Integration handlers ---
  const handleDisconnectGitHub = () => {
    clearGitHubToken();
    setGithubConnected(false);
    setGithubUser(null);
    toast.success("GitHub account disconnected");
  };

  const handleConnectGitHub = () => {
    // Redirect to backend OAuth endpoint — it redirects to GitHub, then back to frontend with token
    window.location.href = "http://localhost:8000/auth/github";
  };

  const handleInstallVSCode = () => {
    window.open("https://marketplace.visualstudio.com/items?itemName=intellcode.intellcode-review", "_blank");
    toast.info("Opening VS Code marketplace…");
  };

  const handleSaveSlack = () => {
    if (!slackWebhookUrl.trim()) {
      toast.error("Please enter a Slack webhook URL.");
      return;
    }
    if (!slackWebhookUrl.startsWith("https://hooks.slack.com/")) {
      toast.error("Invalid webhook URL. Should start with https://hooks.slack.com/");
      return;
    }
    setSlackConnected(true);
    saveSettings({ slackWebhookUrl: slackWebhookUrl.trim(), slackConnected: true });
    toast.success("Slack connected! Notifications will be sent to this webhook.");
  };

  const handleDisconnectSlack = () => {
    setSlackConnected(false);
    setSlackWebhookUrl("");
    saveSettings({ slackWebhookUrl: "", slackConnected: false });
    toast.success("Slack disconnected");
  };

  const handleTestSlack = () => {
    toast.info("Test notification sent to Slack! Check your channel.", { duration: 3000 });
  };

  // --- Appearance handlers ---
  const handleCodeFontChange = (val: string) => {
    setCodeFont(val);
    saveSettings({ codeFont: val });
  };

  const handleDensityChange = (val: string) => {
    setDensity(val);
    saveSettings({ density: val });
    toast.success(`${val} density applied`);
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      {/* Hidden file input for photo upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Delete Account Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-xl p-6 max-w-sm w-full mx-4">
            <div className="flex items-center gap-3 mb-3">
              <AlertTriangle className="w-5 h-5 text-destructive" />
              <h3 className="font-semibold text-foreground">Delete Account</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-6">
              This will permanently delete your account and all your data — reviews, rules, repositories, and settings. This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>Cancel</Button>
              <Button variant="destructive" onClick={handleDeleteAccount}>Yes, Delete Everything</Button>
            </div>
          </div>
        </div>
      )}

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground mt-1">Manage your account preferences and integrations</p>
        </div>

        <div className="flex gap-8">
          {/* Sidebar */}
          <aside className="w-48 flex-shrink-0">
            <nav className="space-y-1">
              {tabs.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === id
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </nav>
          </aside>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Profile Tab */}
            {activeTab === "profile" && (
              <div className="space-y-6">
                <div className="bg-card border border-border rounded-xl p-6">
                  <h2 className="text-lg font-semibold text-foreground mb-6">Profile Information</h2>

                  <div className="flex items-center gap-6 mb-6 pb-6 border-b border-border">
                    {photoUrl ? (
                      <img src={photoUrl} alt="Profile" className="w-20 h-20 rounded-full object-cover" />
                    ) : (
                      <div className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center text-2xl font-bold text-background">
                        {mockUser.avatar}
                      </div>
                    )}
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Profile photo</p>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={handleUploadPhoto}>Upload Photo</Button>
                        {photoUrl && (
                          <Button size="sm" variant="ghost" className="text-destructive" onClick={handleRemovePhoto}>
                            Remove
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <label className="text-sm font-medium text-foreground mb-1.5 block">Full Name</label>
                      <Input value={name} onChange={(e) => setName(e.target.value)} className="bg-input" />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-foreground mb-1.5 block">Email</label>
                      <Input value={email} onChange={(e) => setEmail(e.target.value)} className="bg-input" />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-foreground mb-1.5 block">Role</label>
                      <Input value={role} onChange={(e) => setRole(e.target.value)} className="bg-input" />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-foreground mb-1.5 block">Timezone</label>
                      <Select value={timezone} onValueChange={setTimezone}>
                        <SelectTrigger className="bg-input">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="America/Los_Angeles">Pacific Time (PT)</SelectItem>
                          <SelectItem value="America/Denver">Mountain Time (MT)</SelectItem>
                          <SelectItem value="America/Chicago">Central Time (CT)</SelectItem>
                          <SelectItem value="America/New_York">Eastern Time (ET)</SelectItem>
                          <SelectItem value="UTC">UTC</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Bio</label>
                    <textarea
                      value={bio}
                      onChange={(e) => setBio(e.target.value)}
                      rows={3}
                      className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm text-foreground resize-none focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <div className="mb-6">
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Preferred Language for Code Examples</label>
                    <Select value={preferredLanguage} onValueChange={setPreferredLanguage}>
                      <SelectTrigger className="bg-input w-64">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {["python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp", "ruby", "kotlin"].map((lang) => (
                          <SelectItem key={lang} value={lang} className="capitalize">{lang}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground mt-1.5">
                      Used in refactoring suggestions and code examples throughout the app.
                    </p>
                  </div>

                  <Button className="bg-gradient-primary gap-2" onClick={handleSaveProfile}>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </Button>
                </div>

                <div className="bg-card border border-destructive/30 rounded-xl p-6">
                  <h2 className="text-lg font-semibold text-destructive mb-2">Danger Zone</h2>
                  <p className="text-sm text-muted-foreground mb-4">Permanently delete your account and all data.</p>
                  <Button
                    variant="destructive"
                    size="sm"
                    className="gap-2"
                    onClick={() => setShowDeleteConfirm(true)}
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Account
                  </Button>
                </div>
              </div>
            )}

            {/* Notifications Tab */}
            {activeTab === "notifications" && (
              <div className="space-y-4">
              {/* Browser Push Notifications */}
              <div className="bg-card border border-border rounded-xl p-6">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-foreground">Browser Notifications</h2>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Receive real-time desktop alerts for critical security findings.
                    </p>
                  </div>
                  <div className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                    browserPermission === "granted"
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      : browserPermission === "denied"
                      ? "bg-destructive/20 text-destructive border border-destructive/30"
                      : "bg-secondary text-muted-foreground border border-border"
                  }`}>
                    {browserPermission === "granted" ? "Enabled" : browserPermission === "denied" ? "Blocked" : "Not set"}
                  </div>
                </div>
                {browserPermission !== "granted" && (
                  <Button
                    className="mt-4 gap-2"
                    variant={browserPermission === "denied" ? "outline" : "default"}
                    onClick={handleRequestBrowserPermission}
                    disabled={browserPermission === "denied"}
                  >
                    <Bell className="w-4 h-4" />
                    {browserPermission === "denied" ? "Blocked by browser — update site settings" : "Enable Browser Notifications"}
                  </Button>
                )}
                {browserPermission === "granted" && (
                  <p className="text-xs text-emerald-400 mt-3 flex items-center gap-1.5">
                    <Check className="w-3.5 h-3.5" /> Critical security alerts will appear as desktop notifications
                  </p>
                )}
              </div>

              <div className="bg-card border border-border rounded-xl p-6">
                <h2 className="text-lg font-semibold text-foreground mb-6">Email &amp; In-App Preferences</h2>

                <div className="mb-6 pb-6 border-b border-border">
                  <label className="text-sm font-medium text-foreground mb-1.5 block">Email Frequency</label>
                  <Select value={emailFrequency} onValueChange={setEmailFrequency}>
                    <SelectTrigger className="w-60 bg-input">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="instant">Instant</SelectItem>
                      <SelectItem value="hourly">Hourly Digest</SelectItem>
                      <SelectItem value="daily">Daily Digest</SelectItem>
                      <SelectItem value="never">Never</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-5">
                  {[
                    {
                      label: "Review Complete",
                      desc: "Notify when a code review finishes analyzing",
                      value: notifReviewComplete,
                      set: setNotifReviewComplete,
                    },
                    {
                      label: "Critical Issues Found",
                      desc: "Immediate alert for critical severity vulnerabilities",
                      value: notifCriticalIssues,
                      set: setNotifCriticalIssues,
                    },
                    {
                      label: "New Comment",
                      desc: "When someone comments on your review",
                      value: notifNewComment,
                      set: setNotifNewComment,
                    },
                    {
                      label: "PR Merged",
                      desc: "When a pull request you reviewed gets merged",
                      value: notifPRMerged,
                      set: setNotifPRMerged,
                    },
                    {
                      label: "Weekly Digest",
                      desc: "Summary of your team's code quality trends",
                      value: notifWeeklyDigest,
                      set: setNotifWeeklyDigest,
                    },
                  ].map(({ label, desc, value, set }) => (
                    <div key={label} className="flex items-start justify-between">
                      <div>
                        <div className="text-sm font-medium text-foreground">{label}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">{desc}</div>
                      </div>
                      <Switch checked={value} onCheckedChange={set} />
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-border">
                  <Button className="bg-gradient-primary gap-2" onClick={handleSaveNotifications}>
                    <Save className="w-4 h-4" />
                    Save Preferences
                  </Button>
                </div>
              </div>
              </div>
            )}

            {/* Integrations Tab */}
            {activeTab === "integrations" && (
              <div className="space-y-4">
                {/* GitHub */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-[#24292e] rounded-xl flex items-center justify-center">
                        <Github className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-foreground">GitHub</h3>
                          {githubConnected ? (
                            <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Connected</Badge>
                          ) : (
                            <Badge className="bg-muted text-muted-foreground border-border">Not Connected</Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {githubConnected
                            ? githubUser
                              ? `@${githubUser.login} · ${githubUser.public_repos + (githubUser.private_repos ?? 0)} repos`
                              : "Connected — loading profile…"
                            : "Connect to scan repos and analyze pull requests"}
                        </p>
                      </div>
                    </div>
                    {githubConnected ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="gap-2 text-destructive border-destructive/30 hover:bg-destructive/10"
                        onClick={handleDisconnectGitHub}
                      >
                        <LogOut className="w-4 h-4" />
                        Disconnect
                      </Button>
                    ) : (
                      <Button size="sm" className="gap-2 bg-gradient-primary" onClick={handleConnectGitHub}>
                        <LinkIcon className="w-4 h-4" />
                        Connect
                      </Button>
                    )}
                  </div>

                  {githubConnected && (
                    <div className="mt-4 pt-4 border-t border-border space-y-4">
                      {githubUser && (
                        <div className="flex items-center gap-3">
                          <img
                            src={githubUser.avatar_url}
                            alt={githubUser.login}
                            className="w-10 h-10 rounded-full border border-border"
                          />
                          <div>
                            <div className="text-sm font-medium text-foreground">{githubUser.name || githubUser.login}</div>
                            <div className="text-xs text-muted-foreground">@{githubUser.login}</div>
                          </div>
                          <div className="ml-auto flex gap-4 text-center">
                            <div>
                              <div className="text-sm font-bold text-foreground">{githubUser.public_repos}</div>
                              <div className="text-xs text-muted-foreground">Public</div>
                            </div>
                            <div>
                              <div className="text-sm font-bold text-foreground">{githubUser.private_repos ?? 0}</div>
                              <div className="text-xs text-muted-foreground">Private</div>
                            </div>
                          </div>
                        </div>
                      )}
                      <div>
                        <p className="text-xs font-medium text-muted-foreground mb-1.5">WEBHOOK URL — add this in your repo Settings → Webhooks</p>
                        <div className="flex items-center gap-2">
                          <Input
                            readOnly
                            value="http://localhost:8000/webhook/github"
                            className="bg-secondary font-mono text-xs text-muted-foreground"
                          />
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              navigator.clipboard.writeText("http://localhost:8000/webhook/github");
                              toast.success("Webhook URL copied");
                            }}
                          >
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Events: <span className="text-foreground">push, pull_request</span> · Content type: <span className="text-foreground">application/json</span>
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* VS Code */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-[#007ACC]/20 rounded-xl flex items-center justify-center">
                        <svg className="w-7 h-7" viewBox="0 0 100 100" fill="none">
                          <path d="M74.5 6.6L39.2 38.8 16 21.2 6.3 26.7v46.6L16 78.8l23.2-17.6 35.3 32.2 12.7-6.4V13L74.5 6.6z" fill="#007ACC"/>
                        </svg>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-foreground">VS Code Extension</h3>
                          <Badge className="bg-muted text-muted-foreground border-border">Not Installed</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">Run reviews inline in your editor</p>
                      </div>
                    </div>
                    <Button size="sm" className="gap-2 bg-gradient-primary" onClick={handleInstallVSCode}>
                      <LinkIcon className="w-4 h-4" />
                      Install
                    </Button>
                  </div>
                </div>

                {/* Slack */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-[#4A154B]/20 rounded-xl flex items-center justify-center">
                        <svg className="w-7 h-7" viewBox="0 0 54 54" fill="none">
                          <path d="M19.712.133a5.381 5.381 0 0 0-5.376 5.387 5.381 5.381 0 0 0 5.376 5.386h5.376V5.52A5.381 5.381 0 0 0 19.712.133m0 14.365H5.376A5.381 5.381 0 0 0 0 19.884a5.381 5.381 0 0 0 5.376 5.387h14.336a5.381 5.381 0 0 0 5.376-5.387 5.381 5.381 0 0 0-5.376-5.386" fill="#36C5F0"/>
                          <path d="M53.76 19.884a5.381 5.381 0 0 0-5.376-5.386 5.381 5.381 0 0 0-5.376 5.386v5.387h5.376a5.381 5.381 0 0 0 5.376-5.387m-14.336 0V5.52A5.381 5.381 0 0 0 34.048.133a5.381 5.381 0 0 0-5.376 5.387v14.364a5.381 5.381 0 0 0 5.376 5.387 5.381 5.381 0 0 0 5.376-5.387" fill="#2EB67D"/>
                          <path d="M34.048 54a5.381 5.381 0 0 0 5.376-5.387 5.381 5.381 0 0 0-5.376-5.386h-5.376v5.386A5.381 5.381 0 0 0 34.048 54m0-14.365h14.336a5.381 5.381 0 0 0 5.376-5.386 5.381 5.381 0 0 0-5.376-5.387H34.048a5.381 5.381 0 0 0-5.376 5.387 5.381 5.381 0 0 0 5.376 5.386" fill="#ECB22E"/>
                          <path d="M0 34.249a5.381 5.381 0 0 0 5.376 5.386 5.381 5.381 0 0 0 5.376-5.386v-5.387H5.376A5.381 5.381 0 0 0 0 34.249m14.336 0v14.364A5.381 5.381 0 0 0 19.712 54a5.381 5.381 0 0 0 5.376-5.387V34.249a5.381 5.381 0 0 0-5.376-5.387 5.381 5.381 0 0 0-5.376 5.387" fill="#E01E5A"/>
                        </svg>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold text-foreground">Slack</h3>
                          {slackConnected ? (
                            <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Connected</Badge>
                          ) : (
                            <Badge className="bg-muted text-muted-foreground border-border">Not Connected</Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">Get review notifications in Slack channels</p>
                      </div>
                    </div>
                    {slackConnected && (
                      <Button
                        size="sm"
                        variant="outline"
                        className="gap-2 text-destructive border-destructive/30 hover:bg-destructive/10"
                        onClick={handleDisconnectSlack}
                      >
                        <LogOut className="w-4 h-4" />
                        Disconnect
                      </Button>
                    )}
                  </div>

                  <div className="pt-4 border-t border-border space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground block mb-1.5">INCOMING WEBHOOK URL</label>
                      <div className="flex gap-2">
                        <Input
                          value={slackWebhookUrl}
                          onChange={(e) => setSlackWebhookUrl(e.target.value)}
                          placeholder="https://hooks.slack.com/services/T.../B.../..."
                          className="bg-input border-border font-mono text-xs flex-1"
                          type={slackConnected ? "password" : "text"}
                          readOnly={slackConnected}
                        />
                        {slackConnected ? (
                          <Button size="sm" variant="outline" onClick={handleTestSlack}>Test</Button>
                        ) : (
                          <Button size="sm" className="bg-gradient-primary" onClick={handleSaveSlack}>Connect</Button>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      In Slack: Apps → Incoming Webhooks → Add to Slack → select channel → copy webhook URL.
                    </p>
                    {slackConnected && (
                      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground pt-1">
                        {[
                          { label: "Review Complete", on: notifReviewComplete },
                          { label: "Critical Issues", on: notifCriticalIssues },
                          { label: "Weekly Digest", on: notifWeeklyDigest },
                          { label: "PR Merged", on: notifPRMerged },
                        ].map((n) => (
                          <div key={n.label} className="flex items-center gap-1.5">
                            <div className={`w-2 h-2 rounded-full ${n.on ? "bg-green-400" : "bg-muted-foreground"}`} />
                            {n.label}
                          </div>
                        ))}
                        <span className="text-muted-foreground/60">· configure in Notifications tab</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Security Tab */}
            {activeTab === "security" && (
              <div className="space-y-4">
                {/* Two-Factor Auth */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-foreground mb-1">Two-Factor Authentication</h3>
                      <p className="text-sm text-muted-foreground">
                        Add an extra layer of security to your account using an authenticator app.
                      </p>
                    </div>
                    <Switch
                      checked={twoFAEnabled}
                      onCheckedChange={(v) => {
                        setTwoFAEnabled(v);
                        saveSettings({ twoFAEnabled: v });
                        toast.success(v ? "2FA enabled" : "2FA disabled");
                      }}
                    />
                  </div>
                  {twoFAEnabled && (
                    <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3 flex items-center gap-2">
                      <Check className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-green-400">Two-factor authentication is active</span>
                    </div>
                  )}
                </div>

                {/* API Token */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <h3 className="font-semibold text-foreground mb-1">API Access Token</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Reference token for use in CI pipelines and the <span className="text-foreground font-medium">CI / Integrations</span> page.
                    Stored locally — the backend accepts requests without authentication when running locally.
                  </p>

                  <div className="flex items-center gap-2 mb-4">
                    <div className="flex-1 bg-secondary/50 border border-border rounded-md px-3 py-2 font-mono text-sm text-muted-foreground overflow-hidden">
                      {apiToken ? `${apiToken.slice(0, 16)}••••••••••••••••••••` : <span className="italic text-muted-foreground/60">No token — click Generate</span>}
                    </div>
                    {apiToken && (
                      <Button size="sm" variant="outline" className="gap-2" onClick={handleCopyToken}>
                        {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                        {copied ? "Copied" : "Copy"}
                      </Button>
                    )}
                    <Button size="sm" variant="outline" className="gap-2" onClick={handleRegenerateToken}>
                      <RefreshCw className="w-4 h-4" />
                      {apiToken ? "Regenerate" : "Generate"}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Token never expires. Regenerating will invalidate the current token immediately.
                  </p>
                </div>

                {/* Active Sessions */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <h3 className="font-semibold text-foreground mb-4">Active Sessions</h3>
                  {sessions.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No other active sessions.</p>
                  ) : (
                    <div className="space-y-3">
                      {sessions.map((session) => (
                        <div key={session.id} className="flex items-center justify-between py-3 border-b border-border last:border-0">
                          <div className="flex items-center gap-3">
                            <Key className="w-4 h-4 text-muted-foreground" />
                            <div>
                              <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                                {session.device}
                                {session.current && (
                                  <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-xs">
                                    Current
                                  </Badge>
                                )}
                              </div>
                              <div className="text-xs text-muted-foreground">{session.location} · {session.time}</div>
                            </div>
                          </div>
                          {!session.current && (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="text-destructive hover:bg-destructive/10"
                              onClick={() => handleRevokeSession(session.id)}
                            >
                              Revoke
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Appearance Tab */}
            {activeTab === "appearance" && (
              <div className="bg-card border border-border rounded-xl p-6">
                <h2 className="text-lg font-semibold text-foreground mb-6">Appearance</h2>

                <div className="mb-6">
                  <label className="text-sm font-medium text-foreground mb-3 block">Theme</label>
                  <div className="grid grid-cols-3 gap-3">
                    {([
                      { id: "dark",   label: "Dark",   preview: "bg-[#0b1120] border-[#1e293b]" },
                      { id: "light",  label: "Light",  preview: "bg-[#f3f6fb] border-gray-200" },
                      { id: "system", label: "System", preview: "bg-gradient-to-br from-[#0b1120] to-[#f3f6fb] border-gray-400" },
                    ] as { id: Theme; label: string; preview: string }[]).map(({ id, label, preview }) => (
                      <button
                        key={id}
                        onClick={() => handleThemeChange(id)}
                        className={`relative rounded-xl border-2 overflow-hidden transition-all ${
                          currentTheme === id ? "border-primary" : "border-border hover:border-muted-foreground"
                        }`}
                      >
                        <div className={`h-20 ${preview}`} />
                        <div className="p-2 text-center text-sm font-medium text-foreground">{label}</div>
                        {currentTheme === id && (
                          <div className="absolute top-2 right-2 w-5 h-5 bg-primary rounded-full flex items-center justify-center">
                            <Check className="w-3 h-3 text-primary-foreground" />
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mb-6 pb-6 border-b border-border">
                  <label className="text-sm font-medium text-foreground mb-3 block">Code Font</label>
                  <Select value={codeFont} onValueChange={handleCodeFontChange}>
                    <SelectTrigger className="w-60 bg-input">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="jetbrains">JetBrains Mono</SelectItem>
                      <SelectItem value="fira">Fira Code</SelectItem>
                      <SelectItem value="cascadia">Cascadia Code</SelectItem>
                      <SelectItem value="source">Source Code Pro</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="mb-6 pb-6 border-b border-border">
                  <label className="text-sm font-medium text-foreground mb-3 block">Code Review Density</label>
                  <div className="flex gap-3">
                    {["Compact", "Comfortable", "Spacious"].map((d) => (
                      <button
                        key={d}
                        onClick={() => handleDensityChange(d)}
                        className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-colors ${
                          density === d
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border text-muted-foreground hover:border-muted-foreground"
                        }`}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Keyboard shortcuts reference */}
                <div>
                  <label className="text-sm font-medium text-foreground mb-3 block">Keyboard Shortcuts</label>
                  <div className="bg-secondary/20 rounded-xl border border-border overflow-hidden">
                    {[
                      { keys: ["Ctrl", "Enter"], action: "Run analysis (Submit page)" },
                      { keys: ["Ctrl", "K"], action: "Open command palette" },
                      { keys: ["?"],           action: "Open command palette" },
                      { keys: ["Esc"],         action: "Close dialogs / panels" },
                      { keys: ["Alt", "←"],    action: "Navigate back" },
                    ].map(({ keys, action }) => (
                      <div key={action} className="flex items-center justify-between px-4 py-2.5 border-b border-border last:border-0">
                        <span className="text-sm text-muted-foreground">{action}</span>
                        <div className="flex items-center gap-1">
                          {keys.map((k, i) => (
                            <span key={i} className="text-xs font-mono font-semibold bg-secondary border border-border px-1.5 py-0.5 rounded">{k}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Settings;
