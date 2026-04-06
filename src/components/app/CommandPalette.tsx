import { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  LayoutDashboard, Send, FileText, BarChart2, GitMerge, GitBranch,
  BookOpen, Settings, Users, Code2, Terminal, Shield, Search, X,
  Layers, FileCode2, Zap,
} from "lucide-react";
import { getEntries } from "@/services/reviewHistory";

const COMMANDS = [
  { label: "Dashboard",          path: "/dashboard",    icon: LayoutDashboard, group: "Navigate" },
  { label: "Submit Code",        path: "/submit",       icon: Send,            group: "Navigate" },
  { label: "Batch Analysis",     path: "/batch",        icon: Layers,          group: "Navigate" },
  { label: "Quick Scan",         path: "/submit?mode=quick", icon: Zap,        group: "Actions"  },
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

const SEV_DOT: Record<string, string> = {
  critical: "bg-red-500", high: "bg-orange-400",
  medium: "bg-yellow-400", low: "bg-blue-400", none: "bg-emerald-400",
};

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Load recent file reviews (deduplicated by filename, latest first)
  const recentFiles = useMemo(() => {
    const entries = getEntries();
    const seen = new Set<string>();
    return entries.filter((e) => { if (seen.has(e.filename)) return false; seen.add(e.filename); return true; }).slice(0, 8);
  }, [open]); // refresh on open

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
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

  const filteredCmds = COMMANDS.filter((c) =>
    !query || c.label.toLowerCase().includes(query.toLowerCase()) || c.path.includes(query.toLowerCase())
  );

  const filteredFiles = recentFiles.filter((e) =>
    !query || e.filename.toLowerCase().includes(query.toLowerCase()) || (e.repoName ?? "").toLowerCase().includes(query.toLowerCase())
  );

  // Combined list for keyboard navigation
  const allItems: { type: "cmd" | "file"; idx: number }[] = [
    ...filteredCmds.map((_, i) => ({ type: "cmd" as const, idx: i })),
    ...filteredFiles.map((_, i) => ({ type: "file" as const, idx: i })),
  ];

  const [highlighted, setHighlighted] = useState(0);
  useEffect(() => { setHighlighted(0); }, [query]);

  const go = (path: string) => { setOpen(false); navigate(path); };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlighted((h) => Math.min(h + 1, allItems.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlighted((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      const item = allItems[highlighted];
      if (!item) return;
      if (item.type === "cmd") go(filteredCmds[item.idx].path);
      else go(`/reviews/${filteredFiles[item.idx].id}`);
    }
  };

  if (!open) return null;

  let globalIdx = 0;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] px-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />

      <div className="relative w-full max-w-lg bg-card border border-border rounded-2xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-4 h-4 text-muted-foreground shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Search pages or recent files…"
            className="flex-1 bg-transparent text-foreground text-sm placeholder:text-muted-foreground focus:outline-none"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-muted-foreground hover:text-foreground transition-colors">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <button onClick={() => setOpen(false)} className="text-muted-foreground hover:text-foreground transition-colors ml-1">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[360px] overflow-y-auto py-1.5">
          {allItems.length === 0 ? (
            <div className="px-4 py-6 text-center text-sm text-muted-foreground">No results for "{query}"</div>
          ) : (
            <>
              {/* Pages/commands */}
              {filteredCmds.length > 0 && (
                <>
                  <div className="px-4 py-1.5 text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-wider">Pages</div>
                  {filteredCmds.map((cmd) => {
                    const hi = globalIdx === highlighted;
                    const gi = globalIdx++;
                    const Icon = cmd.icon;
                    return (
                      <button key={cmd.path} onClick={() => go(cmd.path)} onMouseEnter={() => setHighlighted(gi)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors ${hi ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:bg-secondary/40 hover:text-foreground"}`}>
                        <Icon className={`w-4 h-4 shrink-0 ${hi ? "text-primary" : ""}`} />
                        <span className="flex-1">{cmd.label}</span>
                        {cmd.group === "Actions" && <span className="text-[10px] bg-primary/15 text-primary px-1.5 py-0.5 rounded font-medium">Action</span>}
                        <span className="text-xs text-muted-foreground/40 font-mono hidden sm:block">{cmd.path}</span>
                      </button>
                    );
                  })}
                </>
              )}

              {/* Recent files */}
              {filteredFiles.length > 0 && (
                <>
                  <div className="px-4 py-1.5 mt-1 text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-wider border-t border-border/50 pt-2">Recent Files</div>
                  {filteredFiles.map((entry) => {
                    const hi = globalIdx === highlighted;
                    const gi = globalIdx++;
                    return (
                      <button key={entry.id} onClick={() => go(`/reviews/${entry.id}`)} onMouseEnter={() => setHighlighted(gi)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors ${hi ? "bg-primary/15 text-foreground" : "text-muted-foreground hover:bg-secondary/40 hover:text-foreground"}`}>
                        <FileCode2 className={`w-4 h-4 shrink-0 ${hi ? "text-primary" : ""}`} />
                        <div className="flex-1 min-w-0">
                          <span className="truncate block text-foreground text-xs font-medium">{entry.filename}</span>
                          {entry.repoName && <span className="text-[10px] text-muted-foreground truncate block">{entry.repoName}</span>}
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span className={`w-1.5 h-1.5 rounded-full ${SEV_DOT[entry.severity] ?? SEV_DOT.none}`} />
                          <span className="text-xs font-semibold tabular-nums text-foreground">{entry.overallScore}</span>
                        </div>
                      </button>
                    );
                  })}
                </>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-border flex items-center gap-4 text-xs text-muted-foreground/60">
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">↑↓</kbd> navigate</span>
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Enter</kbd> open</span>
          <span><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Esc</kbd> close</span>
          <span className="ml-auto"><kbd className="bg-secondary px-1.5 py-0.5 rounded text-[10px]">Ctrl+K</kbd></span>
        </div>
      </div>
    </div>
  );
}
