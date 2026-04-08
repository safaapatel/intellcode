import { useState, useMemo, useEffect, useCallback } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Copy, Check, Shield, Zap, GitBranch, Terminal, ExternalLink, MessageSquare, Download, FileCode, Activity, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { getSession } from "@/services/auth";
import { getEntries } from "@/services/reviewHistory";

const BASE = import.meta.env.VITE_API_URL || "https://intellcode.onrender.com";

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

// ─── Webhook Event Log ─────────────────────────────────────────────────────────

const WEBHOOK_LOG_KEY = "intellcode_webhook_events";
const MAX_LOG_ENTRIES = 50;

interface WebhookEvent {
  id: string;
  timestamp: string;
  type: "push" | "pull_request" | "analysis" | "error" | "info";
  repo?: string;
  branch?: string;
  score?: number;
  status: "success" | "failure" | "pending" | "info";
  message: string;
  detail?: string;
}

function loadWebhookEvents(): WebhookEvent[] {
  try {
    const raw = localStorage.getItem(WEBHOOK_LOG_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function appendWebhookEvent(event: Omit<WebhookEvent, "id" | "timestamp">) {
  const events = loadWebhookEvents();
  const entry: WebhookEvent = {
    ...event,
    id: Date.now().toString(),
    timestamp: new Date().toISOString(),
  };
  const updated = [entry, ...events].slice(0, MAX_LOG_ENTRIES);
  localStorage.setItem(WEBHOOK_LOG_KEY, JSON.stringify(updated));
  return updated;
}

const STATUS_STYLES: Record<WebhookEvent["status"], string> = {
  success: "text-green-400 bg-green-500/10 border-green-500/30",
  failure: "text-red-400 bg-red-500/10 border-red-500/30",
  pending: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
  info:    "text-blue-400 bg-blue-500/10 border-blue-500/30",
};

const TYPE_LABELS: Record<WebhookEvent["type"], string> = {
  push:         "push",
  pull_request: "pull_request",
  analysis:     "analysis",
  error:        "error",
  info:         "info",
};

function WebhookEventLog({ base }: { base: string }) {
  const [events, setEvents] = useState<WebhookEvent[]>(loadWebhookEvents);
  const [polling, setPolling] = useState(false);

  const clearLog = useCallback(() => {
    localStorage.setItem(WEBHOOK_LOG_KEY, "[]");
    setEvents([]);
  }, []);

  // Poll /stats endpoint every 10s to detect new analyses triggered by webhooks
  useEffect(() => {
    if (!polling) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${base}/stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const updated = appendWebhookEvent({
          type: "analysis",
          status: "info",
          message: `Backend stats: ${data.total_analyses ?? 0} total analyses`,
          detail: `avg_score=${data.avg_score?.toFixed(1) ?? "N/A"}, models_ready=${data.models_ready ?? "?"}`,
          score: data.avg_score ?? undefined,
        });
        setEvents(updated);
      } catch (err) {
        // Backend offline — log it once per poll cycle
      }
    }, 10_000);
    return () => clearInterval(interval);
  }, [polling, base]);

  // Simulate an initial connection check when toggling polling on
  useEffect(() => {
    if (!polling) return;
    (async () => {
      try {
        const res = await fetch(`${base}/health`);
        const data = await res.json();
        const updated = appendWebhookEvent({
          type: "info",
          status: data.status === "ok" ? "success" : "failure",
          message: `Connected to backend — status: ${data.status}`,
          detail: `Models ready: ${Object.values(data.models ?? {}).filter(v => v === "ready").length}`,
        });
        setEvents(updated);
      } catch {
        const updated = appendWebhookEvent({
          type: "error",
          status: "failure",
          message: "Cannot reach backend — is it running?",
          detail: base,
        });
        setEvents(updated);
      }
    })();
  }, [polling, base]);

  return (
    <section className="bg-card border border-border rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <h2 className="font-semibold text-foreground">Webhook Event Log</h2>
          {polling && (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              live
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPolling(p => !p)}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border transition-colors ${
              polling
                ? "bg-green-500/10 border-green-500/30 text-green-400"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            <RefreshCw className={`w-3 h-3 ${polling ? "animate-spin" : ""}`} />
            {polling ? "Stop" : "Start"} polling
          </button>
          <button
            onClick={clearLog}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border border-border text-muted-foreground hover:text-foreground transition-colors"
          >
            <Trash2 className="w-3 h-3" /> Clear
          </button>
        </div>
      </div>

      <p className="text-sm text-muted-foreground mb-4">
        Shows analysis events and backend status checks. When your GitHub webhook fires,
        the backend logs it — click "Start polling" to stream live events from the{" "}
        <code className="text-primary bg-primary/10 px-1 rounded">/stats</code> endpoint.
      </p>

      {events.length === 0 ? (
        <div className="text-center py-10 text-muted-foreground text-sm border border-dashed border-border rounded-xl">
          No events yet. Start polling or push a commit to trigger a webhook.
        </div>
      ) : (
        <div className="space-y-1.5 max-h-80 overflow-y-auto">
          {events.map(ev => (
            <div key={ev.id} className={`flex items-start gap-3 px-3 py-2 rounded-lg border text-xs ${STATUS_STYLES[ev.status]}`}>
              <span className="font-mono text-muted-foreground flex-shrink-0 mt-0.5">
                {new Date(ev.timestamp).toLocaleTimeString()}
              </span>
              <span className="font-mono flex-shrink-0 mt-0.5 opacity-70">[{TYPE_LABELS[ev.type]}]</span>
              <div className="flex-1 min-w-0">
                <span className="font-medium">{ev.message}</span>
                {ev.detail && <span className="ml-2 opacity-60">{ev.detail}</span>}
              </div>
              {ev.score != null && (
                <span className="flex-shrink-0 font-semibold">{ev.score.toFixed(0)}/100</span>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
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
              <span className="ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-500 border border-yellow-500/20">
                Demo only — not validated by backend
              </span>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              Use this token to authenticate API requests from CI pipelines.
              Add it as a secret named <code className="text-primary bg-primary/10 px-1 rounded">INTELLICODE_TOKEN</code> in your repo settings.
            </p>
            <div className="mb-4 flex items-start gap-2 rounded-lg border border-yellow-500/20 bg-yellow-500/5 px-3 py-2.5 text-xs text-yellow-600 dark:text-yellow-400">
              <Shield className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                This token is generated locally and stored in your browser. The backend does not
                currently validate it — set <code className="font-mono">INTELLICODE_API_KEY</code> in
                your server environment and pass it via the <code className="font-mono">X-API-Key</code> header
                for production use.
              </span>
            </div>
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

          {/* PR Inline Review — new workflow */}
          <section className="bg-card border border-border rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-1">
              <MessageSquare className="w-4 h-4 text-primary" />
              <h2 className="font-semibold text-foreground">PR Inline Review (GitHub Actions)</h2>
              <span className="text-xs bg-primary/15 text-primary px-2 py-0.5 rounded-full ml-1">New</span>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              A richer workflow that posts findings as <strong className="text-foreground">inline comments on the exact lines</strong> of each PR, not just a CI pass/fail badge. Security issues, complex functions, and code smells appear directly in the GitHub review UI.
            </p>
            <div className="grid sm:grid-cols-3 gap-3 mb-5">
              {[
                { icon: Shield, label: "Inline security alerts", desc: "On the exact vulnerable line" },
                { icon: Zap, label: "Complexity warnings", desc: "Per-function, in the diff" },
                { icon: GitBranch, label: "Base-branch delta", desc: "Shows score change vs base" },
              ].map(({ icon: Icon, label, desc }) => (
                <div key={label} className="flex items-start gap-2 p-3 bg-secondary/30 border border-border rounded-lg">
                  <Icon className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs font-semibold text-foreground">{label}</p>
                    <p className="text-xs text-muted-foreground">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-sm text-muted-foreground mb-3">
              The workflow file is already included in this repository at{" "}
              <code className="text-primary bg-primary/10 px-1 rounded">.github/workflows/intellicode-review.yml</code>.
              To use it in your own repo, copy both files below:
            </p>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-secondary/40 border border-border rounded-lg">
                <div className="flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm font-mono text-foreground">.github/workflows/intellicode-review.yml</span>
                </div>
                <Button size="sm" variant="outline" onClick={() => {
                  window.open("https://github.com/safaapatel/intellcode/blob/main/.github/workflows/intellicode-review.yml", "_blank");
                }}>
                  <ExternalLink className="w-3.5 h-3.5 mr-1" /> View
                </Button>
              </div>
              <div className="flex items-center justify-between p-3 bg-secondary/40 border border-border rounded-lg">
                <div className="flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm font-mono text-foreground">backend/github_action_review.py</span>
                </div>
                <Button size="sm" variant="outline" onClick={() => {
                  window.open("https://github.com/safaapatel/intellcode/blob/main/backend/github_action_review.py", "_blank");
                }}>
                  <ExternalLink className="w-3.5 h-3.5 mr-1" /> View
                </Button>
              </div>
            </div>
            <div className="mt-4 p-3 bg-secondary/30 border border-border rounded-lg">
              <p className="text-xs font-semibold text-foreground mb-2">Configure per-repo behaviour with <code className="text-primary">.intellicode.yml</code>:</p>
              <pre className="text-xs font-mono text-muted-foreground whitespace-pre">{`threshold: 65          # REQUEST_CHANGES below this score
fail_on_issues: false  # set true to block merges
ignore_paths:
  - "migrations/**"
  - "tests/fixtures/**"
suppress_rules:
  - "insecure_deserialization"  # silence specific rule IDs`}</pre>
              <Button
                size="sm"
                variant="outline"
                className="mt-3"
                onClick={() => {
                  const config = `# IntelliCode Review Configuration\nthreshold: 65\nfail_on_issues: false\nignore_paths:\n  - "migrations/**"\n  - "tests/fixtures/**"\nsuppress_rules: []\nmax_inline_comments: 30\n`;
                  const blob = new Blob([config], { type: "text/yaml" });
                  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: ".intellicode.yml" });
                  a.click();
                  URL.revokeObjectURL(a.href);
                }}
              >
                <Download className="w-3.5 h-3.5 mr-1" /> Download .intellicode.yml
              </Button>
            </div>
          </section>

          {/* Webhook Event Log */}
          <WebhookEventLog base={BASE} />

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
