import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GitMerge, Play, AlertCircle, Loader2, Plus, Minus, FileCode } from "lucide-react";
import { toast } from "sonner";
import { analyzeCode } from "@/services/api";

const SAMPLE_DIFF = `diff --git a/auth.py b/auth.py
index a1b2c3d..e4f5a6b 100644
--- a/auth.py
+++ b/auth.py
@@ -12,8 +12,14 @@ import sqlite3

 API_KEY = "sk-prod-hardcoded-secret"

-def authenticate(username, password):
-    query = f"SELECT * FROM users WHERE user='{username}' AND pass='{password}'"
-    conn = sqlite3.connect("users.db")
-    cursor = conn.cursor()
-    cursor.execute(query)
-    return cursor.fetchone() is not None
+def authenticate(username: str, password: str) -> bool:
+    """Authenticate a user against the database."""
+    conn = sqlite3.connect("users.db")
+    cursor = conn.cursor()
+    cursor.execute(
+        "SELECT 1 FROM users WHERE user=? AND pass=?",
+        (username, password),
+    )
+    return cursor.fetchone() is not None`;

interface DiffLine {
  type: "add" | "remove" | "context" | "header";
  content: string;
  lineNum?: number;
}

function parseDiff(raw: string): { lines: DiffLine[]; added: string[]; filename: string } {
  const lines: DiffLine[] = [];
  const added: string[] = [];
  let filename = "diff.py";
  let lineNum = 0;

  for (const rawLine of raw.split("\n")) {
    if (rawLine.startsWith("+++ b/")) {
      filename = rawLine.slice(6).trim();
      lines.push({ type: "header", content: rawLine });
    } else if (rawLine.startsWith("--- ") || rawLine.startsWith("diff ") || rawLine.startsWith("index ")) {
      lines.push({ type: "header", content: rawLine });
    } else if (rawLine.startsWith("@@")) {
      // Parse @@ -x,y +a,b @@ — extract the + start line
      const m = rawLine.match(/\+(\d+)/);
      lineNum = m ? parseInt(m[1]) - 1 : 0;
      lines.push({ type: "header", content: rawLine });
    } else if (rawLine.startsWith("+") && !rawLine.startsWith("+++")) {
      lineNum++;
      const code = rawLine.slice(1);
      added.push(code);
      lines.push({ type: "add", content: code, lineNum });
    } else if (rawLine.startsWith("-") && !rawLine.startsWith("---")) {
      lines.push({ type: "remove", content: rawLine.slice(1) });
    } else {
      lineNum++;
      lines.push({ type: "context", content: rawLine.slice(1) || rawLine, lineNum });
    }
  }

  return { lines, added, filename };
}

const Diff = () => {
  const navigate = useNavigate();
  const [diffText, setDiffText] = useState(SAMPLE_DIFF);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { lines, added, filename } = parseDiff(diffText);
  const addedCode = added.join("\n");
  const addedCount = lines.filter((l) => l.type === "add").length;
  const removedCount = lines.filter((l) => l.type === "remove").length;

  const handleAnalyze = async () => {
    if (!addedCode.trim()) {
      toast.error("No added lines found in the diff");
      return;
    }
    setAnalyzing(true);
    setError(null);
    try {
      const result = await analyzeCode(addedCode, filename, "python");
      navigate("/reviews/result", { state: { result } });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Analysis failed";
      setError(msg);
      toast.error("Analysis failed", { description: msg });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-6 gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center">
              <GitMerge className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Diff Analysis</h1>
              <p className="text-sm text-muted-foreground">Paste a git diff — only the added lines are analysed</p>
            </div>
          </div>
          <Button
            className="bg-gradient-primary shrink-0"
            onClick={handleAnalyze}
            disabled={analyzing || !addedCode.trim()}
          >
            {analyzing
              ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analysing…</>
              : <><Play className="w-4 h-4 mr-2" />Analyse Diff</>}
          </Button>
        </div>

        {/* Stats bar */}
        {diffText.trim() && (
          <div className="flex items-center gap-4 mb-4 text-sm">
            <div className="flex items-center gap-1.5">
              <FileCode className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">{filename}</span>
            </div>
            <Badge className="bg-green-500/15 text-green-400 border-green-500/25">
              <Plus className="w-3 h-3 mr-1" />{addedCount} added
            </Badge>
            <Badge className="bg-red-500/15 text-red-400 border-red-500/25">
              <Minus className="w-3 h-3 mr-1" />{removedCount} removed
            </Badge>
            <span className="text-muted-foreground">{addedCount} lines will be analysed</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Input */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-foreground">Git Diff Input</label>
            <textarea
              value={diffText}
              onChange={(e) => setDiffText(e.target.value)}
              placeholder={"Paste output of `git diff` or `git diff HEAD~1`…"}
              className="flex-1 min-h-[480px] w-full rounded-xl bg-secondary/30 border border-border text-sm font-mono text-foreground p-4 resize-none focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground"
              spellCheck={false}
            />
            <p className="text-xs text-muted-foreground">
              Tip: run <code className="text-primary bg-primary/10 px-1 rounded">git diff HEAD</code> or <code className="text-primary bg-primary/10 px-1 rounded">git diff main...feature</code> and paste here.
            </p>
          </div>

          {/* Rendered diff */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-foreground">Preview</label>
            <div className="flex-1 min-h-[480px] rounded-xl border border-border bg-secondary/20 overflow-auto">
              {lines.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                  Paste a diff on the left to preview
                </div>
              ) : (
                <table className="w-full text-xs font-mono border-collapse">
                  <tbody>
                    {lines.map((line, i) => (
                      <tr
                        key={i}
                        className={
                          line.type === "add"
                            ? "bg-green-500/10"
                            : line.type === "remove"
                            ? "bg-red-500/10"
                            : line.type === "header"
                            ? "bg-primary/8"
                            : ""
                        }
                      >
                        <td className="w-10 text-right text-muted-foreground/50 pr-3 pl-2 select-none border-r border-border">
                          {line.lineNum ?? ""}
                        </td>
                        <td className="px-3 py-0.5 whitespace-pre">
                          <span
                            className={
                              line.type === "add"
                                ? "text-green-400"
                                : line.type === "remove"
                                ? "text-red-400"
                                : line.type === "header"
                                ? "text-primary/70"
                                : "text-foreground/80"
                            }
                          >
                            {line.type === "add" ? "+" : line.type === "remove" ? "−" : line.type === "header" ? "" : " "}
                          </span>
                          <span
                            className={
                              line.type === "add"
                                ? "text-green-300"
                                : line.type === "remove"
                                ? "text-red-300/70 line-through"
                                : line.type === "header"
                                ? "text-muted-foreground"
                                : "text-foreground/70"
                            }
                          >
                            {line.content}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="mt-4 flex items-start gap-3 bg-destructive/10 border border-destructive/30 rounded-xl p-4 text-sm text-destructive">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </main>
    </div>
  );
};

export default Diff;
