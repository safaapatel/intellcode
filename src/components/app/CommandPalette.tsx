import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  LayoutDashboard, Send, FileText, BarChart2, GitMerge, GitBranch,
  BookOpen, Settings, Users, Code2, Terminal, Shield, Search, X,
} from "lucide-react";

const COMMANDS = [
  { label: "Dashboard",          path: "/dashboard",    icon: LayoutDashboard, group: "Navigate" },
  { label: "Submit Code",        path: "/submit",       icon: Send,            group: "Navigate" },
  { label: "Reviews",            path: "/reviews",      icon: FileText,        group: "Navigate" },
  { label: "Analytics",          path: "/analytics",    icon: BarChart2,       group: "Navigate" },
  { label: "Compare Code",       path: "/compare",      icon: Code2,           group: "Navigate" },
  { label: "Diff Analysis",      path: "/diff",         icon: GitMerge,        group: "Navigate" },
  { label: "Repositories",       path: "/repositories", icon: GitBranch,       group: "Navigate" },
  { label: "Rules",              path: "/rules",        icon: BookOpen,        group: "Navigate" },
  { label: "ML Models",          path: "/models",       icon: Shield,          group: "Navigate" },
  { label: "CI / Integrations",  path: "/ci",           icon: Terminal,        group: "Navigate" },
  { label: "Settings",           path: "/settings",     icon: Settings,        group: "Navigate" },
  { label: "Admin",              path: "/admin",        icon: Users,           group: "Navigate" },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ctrl+K or ? (when not in an input)
      const inInput = ["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "?" && !inInput) {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const filtered = COMMANDS.filter((c) =>
    !query || c.label.toLowerCase().includes(query.toLowerCase()) || c.path.includes(query.toLowerCase())
  );

  const [highlighted, setHighlighted] = useState(0);
  useEffect(() => { setHighlighted(0); }, [query]);

  const go = (path: string) => {
    setOpen(false);
    navigate(path);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlighted((h) => Math.min(h + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlighted((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter" && filtered[highlighted]) {
      go(filtered[highlighted].path);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] px-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />

      {/* Palette */}
      <div className="relative w-full max-w-lg bg-card border border-border rounded-2xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-4 h-4 text-muted-foreground shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Search pages…"
            className="flex-1 bg-transparent text-foreground text-sm placeholder:text-muted-foreground focus:outline-none"
          />
          <button onClick={() => setOpen(false)} className="text-muted-foreground hover:text-foreground transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto py-1.5">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center text-sm text-muted-foreground">No results for "{query}"</div>
          ) : (
            filtered.map((cmd, i) => {
              const Icon = cmd.icon;
              return (
                <button
                  key={cmd.path}
                  onClick={() => go(cmd.path)}
                  onMouseEnter={() => setHighlighted(i)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors ${
                    i === highlighted ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:bg-secondary/40 hover:text-foreground"
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${i === highlighted ? "text-primary" : ""}`} />
                  <span className="flex-1">{cmd.label}</span>
                  <span className="text-xs text-muted-foreground/50 font-mono">{cmd.path}</span>
                </button>
              );
            })
          )}
        </div>

        {/* Footer hint */}
        <div className="px-4 py-2 border-t border-border flex items-center gap-4 text-xs text-muted-foreground/60">
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">↑↓</kbd> navigate</span>
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Enter</kbd> open</span>
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Esc</kbd> close</span>
          <span className="ml-auto"><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Ctrl+K</kbd> or <kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">?</kbd></span>
        </div>
      </div>
    </div>
  );
}
