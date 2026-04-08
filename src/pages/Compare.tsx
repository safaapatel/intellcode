import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { analyzeCodeStream } from "@/services/api";
import type { FullAnalysisResult } from "@/types/analysis";
import { Loader2, GitCompare, ArrowRight, Trophy, CheckCircle2, ExternalLink } from "lucide-react";
import { toast } from "sonner";

// ─── Sample code ──────────────────────────────────────────────────────────────

const SAMPLE_A = `import os

def process_users(users, db):
    results = []
    for user in users:
        query = "SELECT * FROM users WHERE id = " + str(user['id'])
        data = db.execute(query)
        total = 0
        for i in range(len(data)):
            for j in range(len(data)):
                total += data[i]['value'] * data[j]['weight']
        results.append(total)
    return results

secret_key = "hardcoded-key-12345"
api_token  = "sk-prod-abc123"
`;

const SAMPLE_B = `def process_users(users, db):
    """Process user data with secure parameterized queries."""
    results = []
    for user in users:
        data = db.execute(
            "SELECT * FROM users WHERE id = ?",
            (user["id"],)
        )
        total = sum(
            row["value"] * other["weight"]
            for row in data for other in data
        )
        results.append(total)
    return results
`;

// ─── Types ────────────────────────────────────────────────────────────────────

interface Slot {
  code: string;
  filename: string;
  result: FullAnalysisResult | null;
  loading: boolean;
}

// ─── Score circle ─────────────────────────────────────────────────────────────

function ScoreCircle({ score, label, size = 96 }: { score: number; label: string; size?: number }) {
  const r = size / 2 - 7;
  const circ = 2 * Math.PI * r;
  const stroke =
    score >= 80 ? "stroke-emerald-500" : score >= 60 ? "stroke-yellow-500" : "stroke-red-500";
  const text =
    score >= 80 ? "text-emerald-500" : score >= 60 ? "text-yellow-500" : "text-red-500";
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="w-full h-full -rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--border))" strokeWidth="7" />
          <circle
            cx={size / 2} cy={size / 2} r={r} fill="none"
            className={stroke} strokeWidth="7"
            strokeDasharray={`${(score / 100) * circ} ${circ}`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${size >= 100 ? "text-2xl" : "text-xl"} ${text}`}>{score}</span>
          <span className="text-[9px] text-muted-foreground">/100</span>
        </div>
      </div>
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
    </div>
  );
}

// ─── Code editor panel ────────────────────────────────────────────────────────

function EditorPanel({
  slot, label, accent, onChange,
}: {
  slot: Slot;
  label: string;
  accent: string;
  onChange: (patch: Partial<Slot>) => void;
}) {
  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden flex-1 min-w-0 flex flex-col">
      {/* Chrome bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-secondary/60 border-b border-border shrink-0">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
        </div>
        <span className={`text-xs font-bold ml-1 ${accent}`}>{label}</span>
        <input
          value={slot.filename}
          onChange={(e) => onChange({ filename: e.target.value })}
          placeholder="filename.py"
          className="ml-auto bg-transparent font-mono text-xs text-muted-foreground focus:outline-none w-32 text-right border-none"
        />
      </div>
      <textarea
        value={slot.code}
        onChange={(e) => onChange({ code: e.target.value })}
        rows={16}
        spellCheck={false}
        className="w-full flex-1 bg-[hsl(220,26%,4%)] font-mono text-xs text-foreground p-3 resize-none focus:outline-none border-none"
        placeholder="Paste your code here…"
      />
    </div>
  );
}

// ─── Comparison metrics config ────────────────────────────────────────────────

const ROWS: Array<{
  label: string;
  getValue: (r: FullAnalysisResult) => { display: string; num: number };
  higherBetter: boolean;
}> = [
  {
    label: "Overall Score",
    getValue: (r) => ({ display: `${r.overall_score ?? "—"}/100`, num: r.overall_score ?? 0 }),
    higherBetter: true,
  },
  {
    label: "Complexity",
    getValue: (r) => ({ display: `${Math.round(r.complexity?.score ?? 0)}/100`, num: r.complexity?.score ?? 0 }),
    higherBetter: true,
  },
  {
    label: "Security Issues",
    getValue: (r) => ({ display: String(r.security?.summary?.total ?? 0), num: r.security?.summary?.total ?? 0 }),
    higherBetter: false,
  },
  {
    label: "Critical Findings",
    getValue: (r) => ({ display: String(r.security?.summary?.critical ?? 0), num: r.security?.summary?.critical ?? 0 }),
    higherBetter: false,
  },
  {
    label: "Bug Probability",
    getValue: (r) => ({
      display: r.bug_prediction ? `${Math.round(r.bug_prediction.bug_probability * 100)}%` : "—",
      num: Math.round((r.bug_prediction?.bug_probability ?? 0) * 100),
    }),
    higherBetter: false,
  },
  {
    label: "Clone Rate",
    getValue: (r) => ({
      display: r.clones ? `${(r.clones.clone_rate * 100).toFixed(0)}%` : "—",
      num: (r.clones?.clone_rate ?? 0) * 100,
    }),
    higherBetter: false,
  },
  {
    label: "Dead Lines",
    getValue: (r) => ({ display: String(r.dead_code?.dead_line_count ?? 0), num: r.dead_code?.dead_line_count ?? 0 }),
    higherBetter: false,
  },
  {
    label: "Tech Debt (min)",
    getValue: (r) => ({
      display: r.technical_debt ? `${r.technical_debt.total_debt_minutes}m` : "—",
      num: r.technical_debt?.total_debt_minutes ?? 0,
    }),
    higherBetter: false,
  },
  {
    label: "Debt Rating",
    getValue: (r) => {
      const grades: Record<string, number> = { A: 5, B: 4, C: 3, D: 2, E: 1, F: 0 };
      const rating = r.technical_debt?.overall_rating ?? "—";
      return { display: rating, num: grades[rating] ?? 3 };
    },
    higherBetter: true,
  },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

const Compare = () => {
  const navigate = useNavigate();
  const [slotA, setSlotA] = useState<Slot>({
    code: SAMPLE_A, filename: "original.py", result: null, loading: false,
  });
  const [slotB, setSlotB] = useState<Slot>({
    code: SAMPLE_B, filename: "refactored.py", result: null, loading: false,
  });
  const [comparing, setComparing] = useState(false);

  const patchA = (p: Partial<Slot>) => setSlotA((s) => ({ ...s, ...p }));
  const patchB = (p: Partial<Slot>) => setSlotB((s) => ({ ...s, ...p }));

  const runComparison = async () => {
    if (!slotA.code.trim() || !slotB.code.trim()) {
      toast.error("Both editors need code");
      return;
    }
    setComparing(true);
    patchA({ result: null, loading: true });
    patchB({ result: null, loading: true });
    toast.info("Analyzing both versions in parallel…");
    try {
      const [rA, rB] = await Promise.all([
        analyzeCodeStream(slotA.code, slotA.filename, "python", () => {}),
        analyzeCodeStream(slotB.code, slotB.filename, "python", () => {}),
      ]);
      patchA({ result: rA, loading: false });
      patchB({ result: rB, loading: false });
      toast.success("Comparison complete!");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast.error("Analysis failed", { description: msg });
      patchA({ loading: false });
      patchB({ loading: false });
    } finally {
      setComparing(false);
    }
  };

  const hasResults = !!slotA.result && !!slotB.result;
  const scoreA = slotA.result?.overall_score ?? 0;
  const scoreB = slotB.result?.overall_score ?? 0;
  const aWins = scoreA >= scoreB;
  const diff = Math.abs(scoreA - scoreB);

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Breadcrumb */}
        <div className="text-sm text-muted-foreground mb-4">
          <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>
            Dashboard
          </span>
          <span className="mx-2">›</span>
          <span className="text-foreground">Compare</span>
        </div>

        {/* Hero */}
        <div className="relative overflow-hidden rounded-2xl mb-6 border border-primary/20 bg-gradient-to-br from-primary/12 via-primary/5 to-transparent">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-primary/8 rounded-full blur-3xl pointer-events-none" />
          <div className="relative z-10 flex items-center justify-between p-5 sm:p-6 gap-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/15 rounded-xl shrink-0">
                <GitCompare className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">Code Comparison</h1>
                <p className="text-sm text-muted-foreground">
                  Run both versions through all ML models and compare every metric
                </p>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  patchA({ code: SAMPLE_A, filename: "original.py", result: null });
                  patchB({ code: SAMPLE_B, filename: "refactored.py", result: null });
                }}
                disabled={comparing}
              >
                Reset Samples
              </Button>
              <Button
                className="bg-gradient-primary gap-2 min-w-[160px]"
                onClick={runComparison}
                disabled={comparing}
              >
                {comparing ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                ) : (
                  <><GitCompare className="w-4 h-4" /> Run Comparison</>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Editors */}
        <div className="flex flex-col lg:flex-row gap-4 mb-6">
          <EditorPanel slot={slotA} label="Version A" accent="text-primary" onChange={patchA} />

          {/* VS divider */}
          <div className="flex lg:flex-col items-center justify-center gap-2 shrink-0">
            <div className="h-px lg:h-auto lg:w-px flex-1 bg-border" />
            <div className="w-9 h-9 rounded-full bg-secondary border border-border flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">
              vs
            </div>
            <div className="h-px lg:h-auto lg:w-px flex-1 bg-border" />
          </div>

          <EditorPanel slot={slotB} label="Version B" accent="text-emerald-400" onChange={patchB} />
        </div>

        {/* Loading state */}
        {comparing && (
          <div className="bg-card border border-border rounded-xl p-6 mb-6 flex items-center justify-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Running ML models on both versions in parallel…
            </p>
          </div>
        )}

        {/* Results */}
        {hasResults && (
          <div className="space-y-6">
            {/* Winner banner */}
            <div
              className={`rounded-2xl border p-6 ${
                diff === 0
                  ? "border-border bg-secondary/20"
                  : aWins
                  ? "border-primary/25 bg-primary/5"
                  : "border-emerald-500/25 bg-emerald-500/5"
              }`}
            >
              <div className="flex items-center justify-around flex-wrap gap-8">
                {/* Score A */}
                <div className="flex flex-col items-center gap-2">
                  <ScoreCircle score={scoreA} label={slotA.filename} size={110} />
                  {aWins && diff > 0 && (
                    <span className="flex items-center gap-1 text-xs font-semibold text-primary">
                      <Trophy className="w-3.5 h-3.5" /> Winner
                    </span>
                  )}
                </div>

                {/* Diff */}
                <div className="text-center space-y-1">
                  {diff === 0 ? (
                    <p className="text-lg font-bold text-foreground">Tie</p>
                  ) : (
                    <>
                      <p className="text-4xl font-bold text-foreground">
                        {diff}
                      </p>
                      <p className="text-xs text-muted-foreground">point{diff !== 1 ? "s" : ""} difference</p>
                      <p className="text-xs text-muted-foreground font-medium">
                        {aWins ? "Version A" : "Version B"} scores higher
                      </p>
                    </>
                  )}
                </div>

                {/* Score B */}
                <div className="flex flex-col items-center gap-2">
                  <ScoreCircle score={scoreB} label={slotB.filename} size={110} />
                  {!aWins && diff > 0 && (
                    <span className="flex items-center gap-1 text-xs font-semibold text-emerald-400">
                      <Trophy className="w-3.5 h-3.5" /> Winner
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Detailed table */}
            <div className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="px-6 py-4 border-b border-border flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-primary" />
                <h2 className="font-semibold text-foreground">Metric-by-Metric Breakdown</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-muted-foreground bg-secondary/20 border-b border-border">
                      <th className="text-left px-6 py-3 font-medium">Metric</th>
                      <th className="text-center px-6 py-3 font-medium text-primary">
                        Version A · {slotA.filename}
                      </th>
                      <th className="text-center px-6 py-3 font-medium text-emerald-400">
                        Version B · {slotB.filename}
                      </th>
                      <th className="text-center px-6 py-3 font-medium">Better</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ROWS.map((row) => {
                      const vA = row.getValue(slotA.result!);
                      const vB = row.getValue(slotB.result!);
                      const tie = vA.num === vB.num;
                      const rowAWins = !tie && (row.higherBetter ? vA.num > vB.num : vA.num < vB.num);
                      const rowBWins = !tie && !rowAWins;
                      return (
                        <tr
                          key={row.label}
                          className="border-b border-border/40 hover:bg-secondary/10 transition-colors"
                        >
                          <td className="px-6 py-3 text-muted-foreground font-medium">{row.label}</td>
                          <td
                            className={`px-6 py-3 text-center font-semibold ${
                              rowAWins ? "text-primary" : "text-foreground"
                            }`}
                          >
                            {vA.display}
                            {rowAWins && <span className="ml-1 text-primary text-xs">✓</span>}
                          </td>
                          <td
                            className={`px-6 py-3 text-center font-semibold ${
                              rowBWins ? "text-emerald-400" : "text-foreground"
                            }`}
                          >
                            {vB.display}
                            {rowBWins && <span className="ml-1 text-emerald-400 text-xs">✓</span>}
                          </td>
                          <td className="px-6 py-3 text-center">
                            {tie ? (
                              <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-full bg-secondary">
                                Tie
                              </span>
                            ) : (
                              <span
                                className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                                  rowAWins
                                    ? "bg-primary/15 text-primary"
                                    : "bg-emerald-500/15 text-emerald-400"
                                }`}
                              >
                                {rowAWins ? "A" : "B"}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* View full reports */}
            <div className="flex flex-wrap gap-3">
              <Button
                variant="outline"
                className="gap-2"
                onClick={() => navigate(slotA.result._entryId ? `/reviews/${slotA.result._entryId}` : "/reviews/result", { state: { result: slotA.result } })}
              >
                <ExternalLink className="w-4 h-4" />
                Full Report — Version A
              </Button>
              <Button
                variant="outline"
                className="gap-2"
                onClick={() => navigate(slotB.result._entryId ? `/reviews/${slotB.result._entryId}` : "/reviews/result", { state: { result: slotB.result } })}
              >
                <ExternalLink className="w-4 h-4" />
                Full Report — Version B
              </Button>
              <Button variant="outline" onClick={() => navigate("/submit")} className="gap-2">
                <ArrowRight className="w-4 h-4" />
                New Analysis
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Compare;
