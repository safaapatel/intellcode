import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Code2, ChevronDown, ShieldCheck, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { mockUser } from "@/data/mockData";
import { getSession, clearSession } from "@/services/auth";

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
  { name: "Repositories",  path: "/repositories" },
  { name: "Reviews",       path: "/reviews" },
  { name: "Compare",       path: "/compare" },
  { name: "Diff",          path: "/diff" },
  { name: "Analytics",     path: "/analytics" },
  { name: "ML Models",     path: "/models" },
  { name: "Rules",         path: "/rules" },
  { name: "CI / Integrations", path: "/ci" },
  { name: "Settings",      path: "/settings" },
];

export const AppNavigation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const session = getSession();
  const isAdmin = session?.role === "admin";
  const [mobileOpen, setMobileOpen] = useState(false);

  // Reactive profile — updates immediately when Settings saves to localStorage
  const [profile, setProfile] = useState(loadProfileSettings);
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "intellcode_settings") setProfile(loadProfileSettings());
    };
    window.addEventListener("storage", onStorage);
    // Also poll within same tab (storage events don't fire in the same window)
    const id = setInterval(() => setProfile(loadProfileSettings()), 2000);
    return () => { window.removeEventListener("storage", onStorage); clearInterval(id); };
  }, []);

  // Backend health indicator — poll every 15s
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  useEffect(() => {
    const check = () => {
      fetch(`${import.meta.env.VITE_API_URL ?? "http://localhost:8000"}/health`, { signal: AbortSignal.timeout(3000) })
        .then((r) => setBackendUp(r.ok))
        .catch(() => setBackendUp(false));
    };
    check();
    const id = setInterval(check, 15_000);
    return () => clearInterval(id);
  }, []);

  const displayName: string = profile.name ?? mockUser.name;
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
          {/* Backend status dot */}
          {session && backendUp !== null && (
            <div
              className={`hidden sm:flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border ${
                backendUp
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : "bg-red-500/10 text-red-400 border-red-500/20"
              }`}
              title={backendUp ? "ML backend online" : "ML backend offline — start uvicorn on port 8000"}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendUp ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
              {backendUp ? "API Online" : "API Offline"}
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
                  <Link to="/settings">Profile</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link to="/settings">Settings</Link>
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
