import { useState } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { getGitHubToken, isGitHubConnected } from "@/services/github";
import {
  Github, AlertTriangle, CheckCircle, Loader2,
  ExternalLink, MapPin, Shield, Bug, ChevronDown, ChevronRight,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const BASE_URL = import.meta.env.VITE_API_URL || "https://intellcode.onrender.com";

interface FileResult {
  filename: string;
  skipped?: boolean;
  security?: { findings: any[]; summary: any };
  complexity?: any;
  bug_prediction?: any;
  function_risk?: { top_k: any[]; total_functions: number; n_high_risk: number };
}

interface PRAnalysisResult {
  status: string;
  pr_url: string;
  pr_title: string;
  files_analyzed: number;
  comment_urls: string[];
  results: FileResult[];
}

const SEV_COLOR: Record<string, string> = {
  critical: "bg-red-500/10 text-red-500 border-red-500/30",
  high:     "bg-orange-500/10 text-orange-500 border-orange-500/30",
  medium:   "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
  low:      "bg-blue-400/10 text-blue-400 border-blue-400/30",
};

// Summarise what will be posted as a comment preview
function buildCommentPreview(result: PRAnalysisResult): string {
  const real = result.results.filter((r) => !r.skipped);
  const nSec = real.reduce((s, r) => s + (r.security?.findings?.length ?? 0), 0);
  const highBug = real.filter((r) => ["high","critical"].includes(r.bug_prediction?.risk_level ?? "")).map((r) => r.filename);

  let preview = `## IntelliCode Review\n\nAnalyzed **${real.length}** file(s)\n\n`;
  if (nSec) preview += `- **${nSec} security finding(s)** will be posted inline\n`;
  if (highBug.length) preview += `- High bug risk: ${highBug.map(f => `\`${f}\``).join(", ")}\n`;
  if (!nSec && !highBug.length) preview += "- No critical issues found — clean summary comment only\n";

  for (const r of real) {
    const findings = r.security?.findings ?? [];
    const postable = findings.filter((f: any) => !f.false_positive && f.confidence >= 0.45);
    const skipped = findings.length - postable.length;
    if (findings.length > 0) {
      preview += `\n**\`${r.filename}\`**: ${postable.length} inline comment(s)`;
      if (skipped > 0) preview += ` · ${skipped} low-confidence skipped`;
      preview += "\n";
    }
  }

  return preview;
}

export default function PRReview() {
  const navigate = useNavigate();
  const connected = isGitHubConnected();
  const [prUrl, setPrUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [posting, setPosting] = useState(false);
  const [result, setResult] = useState<PRAnalysisResult | null>(null);
  const [posted, setPosted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showPreview, setShowPreview] = useState(false);

  // Step 1: analyze without posting
  async function handleAnalyze() {
    if (!prUrl.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setPosted(false);
    setShowPreview(false);

    try {
      const token = getGitHubToken();
      if (!token) throw new Error("GitHub not connected — go to Settings to connect.");

      const res = await fetch(`${BASE_URL}/github/analyze-pr`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pr_url: prUrl.trim(),
          github_token: token,
          post_comments: false,   // always analyze first, post separately
          max_files: 10,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: any) {
      setError((e as Error).message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // Step 2: post comments after user confirms preview
  async function handlePostComments() {
    if (!prUrl.trim()) return;
    setPosting(true);
    setError(null);

    try {
      const token = getGitHubToken();
      if (!token) throw new Error("GitHub not connected.");

      const res = await fetch(`${BASE_URL}/github/analyze-pr`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pr_url: prUrl.trim(),
          github_token: token,
          post_comments: true,
          max_files: 10,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const posted_result = await res.json();
      setResult(posted_result);
      setPosted(true);
      setShowPreview(false);
    } catch (e: any) {
      setError((e as Error).message ?? "Unknown error");
    } finally {
      setPosting(false);
    }
  }

  function toggle(filename: string) {
    setExpanded((prev) => ({ ...prev, [filename]: !prev[filename] }));
  }

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Github className="w-6 h-6" /> PR Review
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Analyze a GitHub pull request and post findings as review comments.
          </p>
        </div>

        {!connected && (
          <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 p-4 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400">GitHub not connected</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Connect your GitHub account in{" "}
                <button onClick={() => navigate("/settings")} className="underline">Settings</button>{" "}
                to analyze PRs and post comments.
              </p>
            </div>
          </div>
        )}

        <div className="rounded-lg border border-border bg-card p-5 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Pull Request URL</label>
            <Input
              placeholder="https://github.com/owner/repo/pull/123"
              value={prUrl}
              onChange={(e) => setPrUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              disabled={loading || !connected}
            />
          </div>

          <div className="flex items-center justify-end">
            <Button
              onClick={handleAnalyze}
              disabled={loading || !connected || !prUrl.trim()}
            >
              {loading ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analyzing...</>
              ) : (
                <><Github className="w-4 h-4 mr-2" />Analyze PR</>
              )}
            </Button>
          </div>

          <p className="text-xs text-muted-foreground">
            Step 1: analyze your PR. Step 2: review findings, then choose to post comments.
          </p>
        </div>

        {error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-500">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Summary header */}
            <div className="rounded-lg border border-border bg-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-foreground">{result.pr_title}</p>
                  <p className="text-xs text-muted-foreground">{result.pr_url}</p>
                </div>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="flex gap-4 text-sm text-muted-foreground">
                <span>{result.files_analyzed} file(s) analyzed</span>
                {posted && result.comment_urls.length > 0 && (
                  <span className="flex items-center gap-1 text-green-500">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Comments posted to GitHub
                  </span>
                )}
              </div>

              {/* Posted comment links */}
              {posted && result.comment_urls.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {result.comment_urls.map((url, i) => (
                    <a key={i} href={url} target="_blank" rel="noopener noreferrer"
                      className="flex items-center gap-1 text-xs text-blue-500 hover:underline">
                      <ExternalLink className="w-3 h-3" />View comment {i + 1}
                    </a>
                  ))}
                </div>
              )}

              {/* Preview / post controls — only before posting */}
              {!posted && (
                <div className="border-t border-border pt-3 space-y-3">
                  {/* Comment preview */}
                  <button
                    className="flex items-center gap-1.5 text-xs text-primary hover:underline"
                    onClick={() => setShowPreview(v => !v)}
                  >
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showPreview ? "rotate-180" : ""}`} />
                    {showPreview ? "Hide" : "Preview"} what will be posted to GitHub
                  </button>

                  {showPreview && (
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                        Comment preview (markdown)
                      </p>
                      <pre className="text-xs text-foreground/80 whitespace-pre-wrap font-mono leading-relaxed">
                        {buildCommentPreview(result)}
                      </pre>
                      <p className="text-[10px] text-muted-foreground mt-2">
                        Low-confidence and false-positive findings are automatically excluded.
                      </p>
                    </div>
                  )}

                  <Button
                    onClick={handlePostComments}
                    disabled={posting}
                    variant="default"
                    className="w-full"
                  >
                    {posting ? (
                      <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Posting to GitHub...</>
                    ) : (
                      <><Github className="w-4 h-4 mr-2" />Post Comments to PR</>
                    )}
                  </Button>
                  <p className="text-xs text-muted-foreground text-center">
                    Only high-confidence findings will be posted as inline comments.
                  </p>
                </div>
              )}
            </div>

            {/* Per-file results */}
            {result.results.filter((r) => !r.skipped).map((file) => {
              const isOpen = expanded[file.filename] ?? false;
              const secFindings = file.security?.findings ?? [];
              const bugLevel = file.bug_prediction?.risk_level ?? "low";
              const topFunctions = file.function_risk?.top_k ?? [];

              return (
                <div key={file.filename} className="rounded-lg border border-border bg-card overflow-hidden">
                  <button
                    className="w-full flex items-center justify-between p-4 text-left hover:bg-muted/30 transition-colors"
                    onClick={() => toggle(file.filename)}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {isOpen ? <ChevronDown className="w-4 h-4 shrink-0" /> : <ChevronRight className="w-4 h-4 shrink-0" />}
                      <span className="font-mono text-sm truncate">{file.filename}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0 ml-2">
                      {secFindings.length > 0 && (
                        <span className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded border ${SEV_COLOR[secFindings[0]?.severity ?? "low"]}`}>
                          <Shield className="w-3 h-3" />{secFindings.length}
                        </span>
                      )}
                      <span className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded border ${SEV_COLOR[bugLevel]}`}>
                        <Bug className="w-3 h-3" />{bugLevel}
                      </span>
                    </div>
                  </button>

                  {isOpen && (
                    <div className="border-t border-border p-4 space-y-4">
                      {/* Security findings */}
                      {secFindings.length > 0 && (
                        <div>
                          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            Security Findings
                          </h4>
                          <div className="space-y-2">
                            {secFindings.map((f: any, i: number) => (
                              <div key={i} className={`rounded border p-2.5 text-sm ${SEV_COLOR[f.severity] ?? SEV_COLOR.low}`}>
                                <div className="flex items-center justify-between">
                                  <span className="font-medium">{f.title}</span>
                                  <div className="flex items-center gap-1 text-xs">
                                    <MapPin className="w-3 h-3" />L{f.lineno}
                                  </div>
                                </div>
                                <p className="text-xs opacity-80 mt-0.5">{f.description}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* High risk functions */}
                      {topFunctions.length > 0 && (
                        <div>
                          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                            High Risk Functions
                          </h4>
                          <div className="space-y-1.5">
                            {topFunctions.slice(0, 3).map((fn: any) => (
                              <div key={fn.name} className="flex items-center justify-between text-sm rounded border border-border bg-muted/20 px-3 py-2">
                                <div className="flex items-center gap-2">
                                  <span className="font-mono">{fn.name}()</span>
                                  <Badge variant="outline" className={`text-xs ${SEV_COLOR[fn.risk_level]}`}>
                                    {fn.risk_level}
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <MapPin className="w-3 h-3" />L{fn.lineno}–{fn.end_lineno}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Complexity */}
                      {file.complexity && (
                        <div className="flex gap-4 text-xs text-muted-foreground">
                          <span>Complexity score: <strong className="text-foreground">{file.complexity.score}</strong></span>
                          <span>Grade: <strong className="text-foreground">{file.complexity.grade}</strong></span>
                          <span>SLOC: <strong className="text-foreground">{file.complexity.sloc}</strong></span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
