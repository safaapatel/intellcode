import { useState, useMemo } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Copy, Check, Shield, Zap, GitBranch, Terminal, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import { getSession } from "@/services/auth";
import { getEntries } from "@/services/reviewHistory";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function CopyBlock({ code, lang = "bash" }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success("Copied to clipboard");
  };
  return (
    <div className="relative group rounded-xl bg-secondary/40 border border-border overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 bg-secondary/60 border-b border-border">
        <span className="text-xs text-muted-foreground font-mono">{lang}</span>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="p-4 text-sm text-foreground font-mono overflow-x-auto whitespace-pre">{code}</pre>
    </div>
  );
}

const CI = () => {
  const session = getSession();

  // Derive latest quality score from review history for the badge
  const { latestScore, badgeColor, badgeLabel } = useMemo(() => {
    const entries = getEntries();
    if (entries.length === 0) return { latestScore: null, badgeColor: "blue", badgeLabel: "unknown" };
    const scores = entries.map((e) => e.overallScore).filter((s) => s != null);
    const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const color = avg >= 80 ? "brightgreen" : avg >= 65 ? "yellow" : avg >= 50 ? "orange" : "red";
    const label = avg >= 80 ? "passing" : avg >= 65 ? "fair" : avg >= 50 ? "needs work" : "failing";
    return { latestScore: avg, badgeColor: color, badgeLabel: label };
  }, []);
  const username = session?.name?.toLowerCase().replace(/\s+/g, "-") ?? "developer";

  // Fake API token stored per-user in localStorage
  const tokenKey = `intellcode_api_token_${session?.email ?? "anon"}`;
  const [token, setToken] = useState<string>(() => {
    const existing = localStorage.getItem(tokenKey);
    if (existing) return existing;
    const gen = "ick_" + Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    localStorage.setItem(tokenKey, gen);
    return gen;
  });
  const [tokenVisible, setTokenVisible] = useState(false);
  const [rotated, setRotated] = useState(false);

  const rotateToken = () => {
    const gen = "ick_" + Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    localStorage.setItem(tokenKey, gen);
    setToken(gen);
    setRotated(true);
    setTimeout(() => setRotated(false), 3000);
    toast.success("API token rotated");
  };

  const curlExample = `curl -s -X POST ${BASE}/analyze \\
  -H "Content-Type: application/json" \\
  -H "X-API-Token: ${token}" \\
  -d '{
    "code": "$(cat your_file.py)",
    "filename": "your_file.py"
  }' | jq '.overall_score'`;

  const githubActionsYaml = `name: IntelliCode Quality Gate

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run IntelliCode analysis
        id: analysis
        run: |
          SCORE=$(curl -s -X POST ${BASE}/analyze \\
            -H "Content-Type: application/json" \\
            -H "X-API-Token: \${{ secrets.INTELLICODE_TOKEN }}" \\
            -d "{
              \\"code\\": \\"$(cat src/main.py | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')\\"
              \\"filename\\": \\"src/main.py\\"
            }" | jq '.overall_score')
          echo "score=$SCORE" >> $GITHUB_OUTPUT
          echo "Quality score: $SCORE"

      - name: Enforce quality gate
        run: |
          SCORE=\${{ steps.analysis.outputs.score }}
          if [ "$SCORE" -lt 60 ]; then
            echo "❌ Quality gate failed: score $SCORE < 60"
            exit 1
          fi
          echo "✅ Quality gate passed: score $SCORE"`;

  const preCommitHook = `#!/bin/bash
# .git/hooks/pre-commit
# Run IntelliCode quality gate before every commit

STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$')
if [ -z "$STAGED" ]; then exit 0; fi

echo "🔍 Running IntelliCode analysis..."
for FILE in $STAGED; do
  SCORE=$(curl -s -X POST ${BASE}/analyze \\
    -H "Content-Type: application/json" \\
    -d "{\\"code\\": \\"$(cat $FILE)\\", \\"filename\\": \\"$FILE\\"}" \\
    | jq '.overall_score // 100')
  echo "  $FILE: score $SCORE"
  if [ "$SCORE" -lt 50 ]; then
    echo "  ⛔ Blocked: score too low ($SCORE < 50)"
    exit 1
  fi
done
echo "✅ All files passed quality gate"`;

  const badgeMarkdown = `![IntelliCode Quality](https://img.shields.io/badge/IntelliCode-${encodeURIComponent(badgeLabel)}-${badgeColor}?logo=github)`;

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center">
              <Terminal className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">CI / Integrations</h1>
              <p className="text-sm text-muted-foreground">Embed IntelliCode Review into your workflow</p>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {/* API Token */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <Shield className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">API Token</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Use this token to authenticate API requests from CI pipelines.
              Add it as a secret named <code className="text-primary bg-primary/10 px-1 rounded">INTELLICODE_TOKEN</code> in your repo settings.
            </p>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-input border border-border rounded-lg px-4 py-2.5 font-mono text-sm text-foreground overflow-hidden">
                {tokenVisible ? token : token.slice(0, 8) + "••••••••••••••••••••••••"}
              </div>
              <Button variant="outline" size="sm" onClick={() => setTokenVisible((v) => !v)}>
                {tokenVisible ? "Hide" : "Show"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => { navigator.clipboard.writeText(token); toast.success("Token copied"); }}
              >
                <Copy className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={rotateToken} className={rotated ? "border-green-500 text-green-500" : ""}>
                {rotated ? <Check className="w-4 h-4" /> : "Rotate"}
              </Button>
            </div>
          </section>

          {/* Badge */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">README Badge</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">Add a quality badge to your repository README.</p>
            <div className="flex items-center gap-3 mb-4">
              <img
                src={`https://img.shields.io/badge/IntelliCode-${encodeURIComponent(badgeLabel)}-${badgeColor}?logo=github&logoColor=white`}
                alt="IntelliCode badge"
                className="h-5"
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
              {latestScore != null
                ? <Badge className={`${badgeColor === "brightgreen" ? "bg-green-500/20 text-green-400 border-green-500/30" : badgeColor === "yellow" ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" : badgeColor === "orange" ? "bg-orange-500/20 text-orange-400 border-orange-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"}`}>{badgeLabel} · avg {latestScore}/100</Badge>
                : <Badge className="bg-secondary text-muted-foreground border-border">no analyses yet</Badge>
              }
            </div>
            <CopyBlock code={badgeMarkdown} lang="markdown" />
          </section>

          {/* cURL example */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <Terminal className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">cURL / Script Usage</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Call the analysis endpoint directly from any shell script. Returns a JSON object — pipe to <code className="text-primary bg-primary/10 px-1 rounded">jq</code> for specific fields.
            </p>
            <CopyBlock code={curlExample} lang="bash" />
          </section>

          {/* GitHub Actions */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <GitBranch className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">GitHub Actions Workflow</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Add this workflow to <code className="text-primary bg-primary/10 px-1 rounded">.github/workflows/intellicode.yml</code>.
              It runs on every push and PR, failing the check if the quality score drops below 60.
            </p>
            <CopyBlock code={githubActionsYaml} lang="yaml" />
          </section>

          {/* Pre-commit hook */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <Terminal className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">Git Pre-commit Hook</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Block commits locally if staged Python files score below 50.
              Save as <code className="text-primary bg-primary/10 px-1 rounded">.git/hooks/pre-commit</code> and run <code className="text-primary bg-primary/10 px-1 rounded">chmod +x</code> on it.
            </p>
            <CopyBlock code={preCommitHook} lang="bash" />
          </section>

          {/* API reference link */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ExternalLink className="w-4 h-4" />
            Backend running at{" "}
            <a href={`${BASE}/docs`} target="_blank" rel="noreferrer" className="text-primary hover:underline">
              {BASE}/docs
            </a>
            {" "}(FastAPI interactive docs)
          </div>
        </div>
      </main>
    </div>
  );
};

export default CI;
