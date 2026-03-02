import { useState, useEffect } from "react";
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

type SettingsTab = "profile" | "notifications" | "integrations" | "security" | "appearance";

const Settings = () => {
  const [activeTab, setActiveTab] = useState<SettingsTab>("profile");
  const [copied, setCopied] = useState(false);
  const saved = loadSettings();

  // Profile state
  const [name, setName] = useState<string>(saved.name ?? mockUser.name);
  const [email, setEmail] = useState<string>(saved.email ?? mockUser.email);
  const [role, setRole] = useState<string>(saved.role ?? mockUser.role);
  const [bio, setBio] = useState<string>(saved.bio ?? "Software engineer focused on code quality and ML systems.");
  const [timezone, setTimezone] = useState<string>(saved.timezone ?? "America/Los_Angeles");

  // Notification state
  const [notifReviewComplete, setNotifReviewComplete] = useState<boolean>(saved.notifReviewComplete ?? true);
  const [notifCriticalIssues, setNotifCriticalIssues] = useState<boolean>(saved.notifCriticalIssues ?? true);
  const [notifWeeklyDigest, setNotifWeeklyDigest] = useState<boolean>(saved.notifWeeklyDigest ?? false);
  const [notifNewComment, setNotifNewComment] = useState<boolean>(saved.notifNewComment ?? true);
  const [notifPRMerged, setNotifPRMerged] = useState<boolean>(saved.notifPRMerged ?? false);
  const [emailFrequency, setEmailFrequency] = useState<string>(saved.emailFrequency ?? "instant");

  // Security state
  const [twoFAEnabled, setTwoFAEnabled] = useState<boolean>(saved.twoFAEnabled ?? false);
  const [apiToken] = useState<string>("ick_live_f4a8b2d1e9c3a7f6b0e5d2c8a4b1f7e3");

  const tabs: { id: SettingsTab; label: string; icon: typeof User }[] = [
    { id: "profile", label: "Profile", icon: User },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "integrations", label: "Integrations", icon: Github },
    { id: "security", label: "Security", icon: Shield },
    { id: "appearance", label: "Appearance", icon: Palette },
  ];

  const handleSaveProfile = () => {
    saveSettings({ name, email, role, bio, timezone });
    toast.success("Profile updated successfully");
  };

  const handleSaveNotifications = () => {
    saveSettings({ notifReviewComplete, notifCriticalIssues, notifWeeklyDigest, notifNewComment, notifPRMerged, emailFrequency });
    toast.success("Notification preferences saved");
  };

  const handleSaveSecurity = () => {
    saveSettings({ twoFAEnabled });
    toast.success("Security settings saved");
  };

  const handleCopyToken = () => {
    setCopied(true);
    toast.success("API token copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRegenerateToken = () => {
    toast.success("New API token generated");
  };

  const handleDisconnectGitHub = () => {
    toast.error("GitHub account disconnected");
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

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
                    <div className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center text-2xl font-bold text-background">
                      {mockUser.avatar}
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Profile photo</p>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">Upload Photo</Button>
                        <Button size="sm" variant="ghost" className="text-destructive">Remove</Button>
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

                  <div className="mb-6">
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Bio</label>
                    <textarea
                      value={bio}
                      onChange={(e) => setBio(e.target.value)}
                      rows={3}
                      className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm text-foreground resize-none focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <Button className="bg-gradient-primary gap-2" onClick={handleSaveProfile}>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </Button>
                </div>

                <div className="bg-card border border-destructive/30 rounded-xl p-6">
                  <h2 className="text-lg font-semibold text-destructive mb-2">Danger Zone</h2>
                  <p className="text-sm text-muted-foreground mb-4">Permanently delete your account and all data.</p>
                  <Button variant="destructive" size="sm" className="gap-2">
                    <Trash2 className="w-4 h-4" />
                    Delete Account
                  </Button>
                </div>
              </div>
            )}

            {/* Notifications Tab */}
            {activeTab === "notifications" && (
              <div className="bg-card border border-border rounded-xl p-6">
                <h2 className="text-lg font-semibold text-foreground mb-6">Notification Preferences</h2>

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
                  <Button
                    className="bg-gradient-primary gap-2"
                    onClick={handleSaveNotifications}
                  >
                    <Save className="w-4 h-4" />
                    Save Preferences
                  </Button>
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
                          <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Connected</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">@safaa-patel · 4 repositories connected</p>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-2 text-destructive border-destructive/30 hover:bg-destructive/10"
                      onClick={handleDisconnectGitHub}
                    >
                      <LogOut className="w-4 h-4" />
                      Disconnect
                    </Button>
                  </div>

                  <div className="mt-4 pt-4 border-t border-border grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-xl font-bold text-foreground">4</div>
                      <div className="text-xs text-muted-foreground">Repositories</div>
                    </div>
                    <div>
                      <div className="text-xl font-bold text-foreground">147</div>
                      <div className="text-xs text-muted-foreground">Reviews Run</div>
                    </div>
                    <div>
                      <div className="text-xl font-bold text-foreground">23</div>
                      <div className="text-xs text-muted-foreground">PRs Analyzed</div>
                    </div>
                  </div>
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
                    <Button size="sm" className="gap-2 bg-gradient-primary">
                      <LinkIcon className="w-4 h-4" />
                      Install
                    </Button>
                  </div>
                </div>

                {/* Slack */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <div className="flex items-start justify-between">
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
                          <Badge className="bg-muted text-muted-foreground border-border">Not Connected</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">Get review notifications in Slack channels</p>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="gap-2">
                      <LinkIcon className="w-4 h-4" />
                      Connect
                    </Button>
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
                    Use this token to authenticate API requests from your CI/CD pipeline.
                  </p>

                  <div className="flex items-center gap-2 mb-4">
                    <div className="flex-1 bg-secondary/50 border border-border rounded-md px-3 py-2 font-mono text-sm text-muted-foreground overflow-hidden">
                      {apiToken.slice(0, 16)}••••••••••••••••••••
                    </div>
                    <Button size="sm" variant="outline" className="gap-2" onClick={handleCopyToken}>
                      {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                      {copied ? "Copied" : "Copy"}
                    </Button>
                    <Button size="sm" variant="outline" className="gap-2" onClick={handleRegenerateToken}>
                      <RefreshCw className="w-4 h-4" />
                      Regenerate
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Token never expires. Regenerating will invalidate the current token immediately.
                  </p>
                </div>

                {/* Active Sessions */}
                <div className="bg-card border border-border rounded-xl p-6">
                  <h3 className="font-semibold text-foreground mb-4">Active Sessions</h3>
                  <div className="space-y-3">
                    {[
                      { device: "Chrome on Windows 11", location: "Reno, NV", current: true, time: "Now" },
                      { device: "Safari on iPhone 15", location: "Reno, NV", current: false, time: "2 hours ago" },
                    ].map((session) => (
                      <div key={session.device} className="flex items-center justify-between py-3 border-b border-border last:border-0">
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
                            onClick={() => toast.success("Session revoked")}
                          >
                            Revoke
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
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
                    {[
                      { id: "dark", label: "Dark", preview: "bg-[#0f172a] border-[#1e293b]" },
                      { id: "light", label: "Light", preview: "bg-white border-gray-200" },
                      { id: "system", label: "System", preview: "bg-gradient-to-br from-[#0f172a] to-white border-gray-400" },
                    ].map(({ id, label, preview }) => (
                      <button
                        key={id}
                        onClick={() => {
                          if (id !== "dark") toast.info(`${label} theme coming soon`);
                        }}
                        className={`relative rounded-xl border-2 overflow-hidden transition-all ${
                          id === "dark" ? "border-primary" : "border-border hover:border-muted-foreground"
                        }`}
                      >
                        <div className={`h-20 ${preview}`} />
                        <div className="p-2 text-center text-sm font-medium text-foreground">
                          {label}
                        </div>
                        {id === "dark" && (
                          <div className="absolute top-2 right-2 w-5 h-5 bg-primary rounded-full flex items-center justify-center">
                            <Check className="w-3 h-3 text-background" />
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mb-6 pb-6 border-b border-border">
                  <label className="text-sm font-medium text-foreground mb-3 block">Code Font</label>
                  <Select defaultValue="jetbrains">
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

                <div>
                  <label className="text-sm font-medium text-foreground mb-3 block">Code Review Density</label>
                  <div className="flex gap-3">
                    {["Compact", "Comfortable", "Spacious"].map((density) => (
                      <button
                        key={density}
                        onClick={() => toast.info(`${density} density selected`)}
                        className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-colors ${
                          density === "Comfortable"
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border text-muted-foreground hover:border-muted-foreground"
                        }`}
                      >
                        {density}
                      </button>
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
