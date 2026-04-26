import { useState, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Code2, ChevronDown, ShieldCheck, Menu, X, Bell, Sun, Moon } from "lucide-react";
import { getTheme, applyTheme, type Theme } from "@/lib/theme";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { getSession, clearSession } from "@/services/auth";
import { getEntries, type HistoryEntry } from "@/services/reviewHistory";

const NOTIF_KEY = "intellcode_notif_seen";

function getUnreadNotifs(): HistoryEntry[] {
  const lastSeen = Number(localStorage.getItem(NOTIF_KEY) ?? 0);
  return getEntries()
    .filter(
      (e) =>
        (e.severity === "critical" || e.severity === "high") &&
        new Date(e.submittedAt).getTime() > lastSeen
    )
    .slice(0, 10);
}

function markNotifsRead() {
  localStorage.setItem(NOTIF_KEY, String(Date.now()));
}

function loadProfileSettings() {
  try {
    const raw = localStorage.getItem("intellcode_settings");
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

const baseNavLinks = [
  { name: "Dashboard",     path: "/dashboard" },
  { name: "Submit",        path: "/submit" },
  { name: "Batch",         path: "/batch" },
  { name: "Repositories",  path: "/repositories" },
  { name: "PR Review",     path: "/pr-review" },
  { name: "Reviews",       path: "/reviews" },
  { name: "Compare",       path: "/compare" },
  { name: "Analytics",     path: "/analytics" },
  { name: "Rules",         path: "/rules" },
  { name: "Settings",      path: "/settings" },
];

export const AppNavigation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const session = getSession();
  const isAdmin = session?.role === "admin";
  const [mobileOpen, setMobileOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [unread, setUnread] = useState<HistoryEntry[]>([]);
  const notifRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const refresh = () => setUnread(getUnreadNotifs());
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, []);

  // Close notif panel on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setNotifOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleOpenNotifs = () => {
    setNotifOpen((v) => !v);
    markNotifsRead();
    setTimeout(() => setUnread([]), 300);
  };

  // Reactive profile — updates immediately when Settings saves to localStorage
  const [profile, setProfile] = useState(loadProfileSettings);
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "intellcode_settings") setProfile(loadProfileSettings());
    };
    window.addEventListener("storage", onStorage);
    return () => { window.removeEventListener("storage", onStorage); };
  }, []);

  // Theme toggle
  const [theme, setTheme] = useState<Theme>(getTheme);
  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  };

  // Backend health indicator
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const failCount = useRef(0);
  useEffect(() => {
    const BASE = import.meta.env.VITE_API_URL ?? "https://intellcode.onrender.com";
    const check = async () => {
      try {
        // 60s timeout — Render free tier cold start can take 50s+
        const ac = new AbortController();
        const t = setTimeout(() => ac.abort(), 60000);
        const r = await fetch(`${BASE}/health`, { signal: ac.signal });
        clearTimeout(t);
        failCount.current = 0;
        setBackendUp(r.ok);
      } catch {
        failCount.current++;
        if (failCount.current >= 3) setBackendUp(false);
      }
    };
    check();
    const id = setInterval(check, 20_000);
    return () => clearInterval(id);
  }, []);

  const displayName: string = profile.name ?? getSession()?.name ?? "User";
  const photoUrl: string = profile.photoUrl ?? "";
  const initials = displayName
    .split(" ")
    .filter(Boolean)
    .map((w: string) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase() || "?";

  const navLinks = isAdmin
    ? [...baseNavLinks, { name: "Admin", path: "/admin" }]
    : baseNavLinks;

  const handleSignOut = () => {
    clearSession();
    navigate("/login");
  };

  return (
    <nav className="sticky top-0 z-50 bg-card/95 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to={session ? "/dashboard" : "/"} className="flex items-center gap-2">
            <div className="p-1.5 bg-gradient-primary rounded-lg">
              <Code2 className="w-5 h-5 text-background" />
            </div>
            <span className="text-lg font-bold text-foreground">IntelliCode Review</span>
          </Link>

          {session && (
            <div className="hidden md:flex items-center gap-0.5">
              {navLinks.map((link) => {
                const isActive = link.path === "/dashboard"
                  ? location.pathname === link.path
                  : location.pathname.startsWith(link.path);
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`relative px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                      isActive
                        ? "text-primary bg-primary/10"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                    }`}
                  >
                    {link.path === "/admin" && <ShieldCheck className="w-3.5 h-3.5" />}
                    {link.name}
                    {isActive && (
                      <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-primary rounded-full" />
                    )}
                  </Link>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Notification bell */}
          {session && (
            <div className="relative" ref={notifRef}>
              <button
                onClick={handleOpenNotifs}
                className="relative p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                title="Notifications"
              >
                <Bell className="w-4 h-4" />
                {unread.length > 0 && (
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                )}
              </button>

              {notifOpen && (
                <div className="absolute right-0 top-10 w-80 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                    <span className="text-sm font-semibold text-foreground">Notifications</span>
                    <button onClick={() => setNotifOpen(false)} className="text-muted-foreground hover:text-foreground">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  {(() => {
                    const all = getEntries().filter((e) => e.severity === "critical" || e.severity === "high");
                    const recent = all.slice(0, 8);
                    const hiddenCount = all.length - recent.length;
                    return recent.length === 0 ? (
                      <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                        No critical or high-severity findings
                      </div>
                    ) : (
                      <div className="divide-y divide-border max-h-72 overflow-y-auto">
                        {recent.map((e) => (
                          <button
                            key={e.id}
                            className="w-full text-left px-4 py-3 hover:bg-secondary/50 transition-colors flex items-start gap-3"
                            onClick={() => { navigate(`/reviews/${e.id}`); setNotifOpen(false); }}
                          >
                            <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${e.severity === "critical" ? "bg-red-500" : "bg-orange-400"}`} />
                            <div className="min-w-0">
                              <p className="text-xs font-medium text-foreground truncate">{e.filename}</p>
                              <p className="text-xs text-muted-foreground">
                                {e.severity === "critical" ? "Critical" : "High"} severity · {e.issueCount} issue{e.issueCount !== 1 ? "s" : ""}
                              </p>
                            </div>
                          </button>
                        ))}
                        {hiddenCount > 0 && (
                          <div className="px-4 py-2 text-center text-xs text-muted-foreground">
                            +{hiddenCount} more — <button className="text-primary hover:underline" onClick={() => { navigate("/reviews"); setNotifOpen(false); }}>view all</button>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                  <div className="border-t border-border px-4 py-2">
                    <button
                      className="text-xs text-primary hover:underline"
                      onClick={() => { navigate("/reviews"); setNotifOpen(false); }}
                    >
                      View all reviews →
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Backend status dot */}
          {session && (
            <div
              className={`hidden sm:flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border transition-all ${
                backendUp === true
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : backendUp === false
                  ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
                  : "bg-secondary text-muted-foreground border-border"
              }`}
              title={
                backendUp === true
                  ? "ML backend online"
                  : backendUp === false
                  ? "Backend may be warming up (Render free tier — retry in ~30s)"
                  : "Checking backend…"
              }
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendUp === true ? "bg-emerald-400 animate-pulse" : backendUp === false ? "bg-yellow-400" : "bg-muted-foreground"}`} />
              {backendUp === true ? "API Online" : backendUp === false ? "Warming Up" : "Checking…"}
            </div>
          )}

          {/* Mobile hamburger — only when logged in */}
          {session && (
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden"
              onClick={() => setMobileOpen((v) => !v)}
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          )}

          {session ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="flex items-center gap-2">
                  {photoUrl ? (
                    <img src={photoUrl} alt={displayName} className="w-8 h-8 rounded-full object-cover" />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center text-sm font-semibold text-background">
                      {initials}
                    </div>
                  )}
                  <div className="hidden sm:flex flex-col items-start">
                    <span className="text-sm leading-none">{displayName}</span>
                    {session.role && (
                      <span className="text-[10px] text-muted-foreground capitalize">{session.role}</span>
                    )}
                  </div>
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem asChild>
                  <Link to="/settings">Profile &amp; Settings</Link>
                </DropdownMenuItem>
                {isAdmin && (
                  <DropdownMenuItem asChild>
                    <Link to="/admin" className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4" />
                      User Management
                    </Link>
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive cursor-pointer"
                  onClick={handleSignOut}
                >
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <>
              <Button variant="ghost" asChild>
                <Link to="/login">Sign In</Link>
              </Button>
              <Button className="bg-gradient-primary" asChild>
                <Link to="/login">Get Started</Link>
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Mobile menu */}
      {session && mobileOpen && (
        <div className="md:hidden border-t border-border bg-card px-4 py-3 space-y-1">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              onClick={() => setMobileOpen(false)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                location.pathname === link.path
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              }`}
            >
              {link.path === "/admin" && <ShieldCheck className="w-3.5 h-3.5" />}
              {link.name}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
};
