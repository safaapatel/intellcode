import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { X, Keyboard } from "lucide-react";

const SHORTCUTS = [
  {
    group: "Navigation",
    items: [
      { keys: ["G", "D"], description: "Go to Dashboard" },
      { keys: ["G", "S"], description: "Go to Submit / New Analysis" },
      { keys: ["G", "R"], description: "Go to Reviews" },
      { keys: ["G", "A"], description: "Go to Analytics" },
      { keys: ["G", "B"], description: "Go to Batch Analysis" },
      { keys: ["G", "M"], description: "Go to Models" },
    ],
  },
  {
    group: "Actions",
    items: [
      { keys: ["Ctrl", "K"], description: "Open command palette" },
      { keys: ["?"], description: "Show keyboard shortcuts" },
      { keys: ["Escape"], description: "Close modal / palette" },
    ],
  },
  {
    group: "Reviews",
    items: [
      { keys: ["Tab"], description: "Cycle through result tabs" },
      { keys: ["Ctrl", "E"], description: "Export report" },
      { keys: ["Ctrl", "P"], description: "Print / save as PDF" },
    ],
  },
];

function Key({ k }: { k: string }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[26px] h-[22px] px-1.5 bg-secondary border border-border rounded text-[11px] font-mono font-semibold text-foreground shadow-sm">
      {k}
    </kbd>
  );
}

export function KeyboardShortcuts() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    let gPressed = false;
    let gTimer: ReturnType<typeof setTimeout> | null = null;

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      const isInput = tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).isContentEditable;

      // ? to open shortcuts (not in input)
      if (e.key === "?" && !isInput) {
        e.preventDefault();
        setOpen((o) => !o);
        return;
      }

      // Escape closes
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }

      if (isInput) return;

      // G + letter navigation chords
      if (e.key === "g" || e.key === "G") {
        gPressed = true;
        if (gTimer) clearTimeout(gTimer);
        gTimer = setTimeout(() => { gPressed = false; }, 1000);
        return;
      }

      if (gPressed) {
        gPressed = false;
        if (gTimer) clearTimeout(gTimer);
        const dest: Record<string, string> = {
          d: "/dashboard", s: "/submit", r: "/reviews",
          a: "/analytics", b: "/batch", m: "/models",
        };
        const key = e.key.toLowerCase();
        if (dest[key]) {
          e.preventDefault();
          navigate(dest[key]);
        }
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      if (gTimer) clearTimeout(gTimer);
    };
  }, [navigate]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={() => setOpen(false)}
    >
      <div
        className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-primary" />
            <span className="font-semibold text-sm text-foreground">Keyboard Shortcuts</span>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-5 max-h-[70vh] overflow-y-auto">
          {SHORTCUTS.map((group) => (
            <div key={group.group}>
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                {group.group}
              </p>
              <div className="space-y-1">
                {group.items.map((item) => (
                  <div key={item.description} className="flex items-center justify-between py-1.5 px-3 rounded-lg hover:bg-secondary/40 transition-colors">
                    <span className="text-sm text-foreground">{item.description}</span>
                    <div className="flex items-center gap-1 shrink-0">
                      {item.keys.map((k, i) => (
                        <span key={i} className="flex items-center gap-1">
                          <Key k={k} />
                          {i < item.keys.length - 1 && <span className="text-muted-foreground text-xs">+</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer hint */}
        <div className="px-5 py-3 border-t border-border bg-secondary/20 text-center">
          <span className="text-xs text-muted-foreground">
            Press <Key k="?" /> anywhere to toggle this panel
          </span>
        </div>
      </div>
    </div>
  );
}
