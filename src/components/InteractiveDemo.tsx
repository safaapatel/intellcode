import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Play, AlertTriangle, CheckCircle2, XCircle, TrendingUp, ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { analyzeCode } from "@/services/api";

const DEMO_CODE = `import os

def process_user_data(user_input, db, config, flag, mode):
    # Security: SQL injection via string concatenation
    query = "SELECT * FROM users WHERE id = " + user_input
    result = db.execute(query)

    # Hardcoded secret
    api_key = "sk_live_1234567890abcdef"
    password = "admin123"

    # Code smell: deep nesting & high cyclomatic complexity
    output = []
    for item in result:
        if item:
            if config:
                if flag:
                    if mode == "fast":
                        output.append(item * 2)
                    else:
                        output.append(item)
                else:
                    output.append(0)
            else:
                output.append(-1)

    # Dead code: unused variable
    unused_var = os.getenv("SECRET")

    return output`;

interface DemoIssue {
  severity: "Critical" | "High" | "Medium" | "Low";
  category: string;
  title: string;
  line: number;
  confidence: number;
  detected_by: string;
}

const FALLBACK_ISSUES: DemoIssue[] = [
  { severity: "Critical", category: "Security", title: "SQL Injection via string concatenation", line: 5, confidence: 98, detected_by: "Pattern Scanner + ML" },
  { severity: "High", category: "Security", title: "Hardcoded API key detected", line: 9, confidence: 96, detected_by: "Pattern Scanner" },
  { severity: "High", category: "Security", title: "Hardcoded password", line: 10, confidence: 94, detected_by: "Pattern Scanner" },
  { severity: "Medium", category: "Complexity", title: "Cyclomatic complexity 6 — deep nesting", line: 14, confidence: 89, detected_by: "Static Analysis" },
  { severity: "Low", category: "Dead Code", title: "Unused variable: unused_var", line: 26, confidence: 82, detected_by: "Static Analysis" },
];

export const InteractiveDemo = () => {
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState("code");
  const [overallScore, setOverallScore] = useState(62);
  const [analysisTime, setAnalysisTime] = useState("—");
  const [issues, setIssues] = useState<DemoIssue[]>(FALLBACK_ISSUES);
  const [hasRun, setHasRun] = useState(false);
  const [isFallback, setIsFallback] = useState(false);

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    const start = Date.now();
    toast.loading("Running static analysis...", { id: "demo" });

    try {
      const result = await analyzeCode(DEMO_CODE, "demo_snippet.py", "python");
      const elapsed = ((Date.now() - start) / 1000).toFixed(1) + "s";
      setAnalysisTime(elapsed);
      setOverallScore(result.overall_score ?? 62);

      const realIssues: DemoIssue[] = [];
      for (const f of result.security?.findings ?? []) {
        realIssues.push({
          severity: f.severity === "critical" ? "Critical" : f.severity === "high" ? "High" : f.severity === "medium" ? "Medium" : "Low",
          category: "Security",
          title: f.title ?? f.vuln_type,
          line: f.lineno ?? 1,
          confidence: Math.round((f.confidence ?? 0.9) * 100),
          detected_by: "Pattern Scanner + ML",
        });
      }
      if ((result.complexity?.cyclomatic ?? 0) > 3) {
        realIssues.push({
          severity: "Medium",
          category: "Complexity",
          title: `Cyclomatic complexity ${result.complexity.cyclomatic} — consider refactoring`,
          line: 3,
          confidence: 92,
          detected_by: "Static Analysis",
        });
      }
      if (result.bug_prediction?.risk_level === "high" || result.bug_prediction?.risk_level === "critical") {
        realIssues.push({
          severity: result.bug_prediction.risk_level === "critical" ? "Critical" : "High",
          category: "Bug Risk",
          title: `Bug probability: ${Math.round(result.bug_prediction.bug_probability * 100)}%`,
          line: 1,
          confidence: Math.round(result.bug_prediction.confidence * 100),
          detected_by: "ML Bug Predictor",
        });
      }
      if (result.dead_code?.issues?.length) {
        realIssues.push({
          severity: "Low",
          category: "Dead Code",
          title: `${result.dead_code.issues.length} dead code issue(s) found`,
          line: result.dead_code.issues[0].start_line,
          confidence: 85,
          detected_by: "Static Analysis",
        });
      }
      if (realIssues.length > 0) setIssues(realIssues.slice(0, 5));
      setIsFallback(false);
      toast.success("Analysis complete!", { id: "demo" });
    } catch {
      // Backend offline — show fallback results with clear indicator
      setAnalysisTime("2.4s");
      setIssues(FALLBACK_ISSUES);
      setIsFallback(true);
      toast.info("Showing sample results — start the backend for live ML analysis", { id: "demo" });
    }

    setIsAnalyzing(false);
    setHasRun(true);
    setActiveTab("results");
  };

  const SEV_ICON: Record<string, JSX.Element> = {
    Critical: <XCircle className="w-5 h-5 text-destructive" />,
    High: <AlertTriangle className="w-5 h-5 text-warning" />,
    Medium: <TrendingUp className="w-5 h-5 text-primary" />,
    Low: <CheckCircle2 className="w-5 h-5 text-muted-foreground" />,
  };

  return (
    <section id="demo" className="py-24 bg-secondary/50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-accent bg-clip-text text-transparent">
            See It In Action
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Watch IntelliCode Review analyze real code with static analysis and ML
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <Card className="p-8 bg-card border-primary/20">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="code">Code Sample</TabsTrigger>
                <TabsTrigger value="results" disabled={!hasRun}>
                  Analysis Results
                </TabsTrigger>
              </TabsList>

              <TabsContent value="code" className="space-y-6">
                <div className="bg-secondary rounded-lg p-6 font-mono text-sm overflow-x-auto">
                  <pre className="text-foreground">{DEMO_CODE}</pre>
                </div>

                <div className="flex justify-center">
                  <Button
                    size="lg"
                    onClick={runAnalysis}
                    disabled={isAnalyzing}
                    className="bg-gradient-primary hover:shadow-glow-primary"
                  >
                    {isAnalyzing ? (
                      <>Analyzing…</>
                    ) : (
                      <>
                        <Play className="mr-2 w-5 h-5" />
                        Run Analysis
                      </>
                    )}
                  </Button>
                </div>
              </TabsContent>

              <TabsContent value="results" className="space-y-6">
                {isFallback && (
                  <div className="flex items-center gap-2 px-3 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-xs text-yellow-500">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                    Sample results — backend offline. Run the FastAPI server to see live ML analysis.
                  </div>
                )}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-warning mb-1">{overallScore}/100</div>
                    <div className="text-sm text-muted-foreground">Quality Score</div>
                  </Card>
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-destructive mb-1">{issues.length}</div>
                    <div className="text-sm text-muted-foreground">Issues Found</div>
                  </Card>
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-accent mb-1">{analysisTime}</div>
                    <div className="text-sm text-muted-foreground">Analysis Time</div>
                  </Card>
                </div>

                <div className="space-y-3">
                  {issues.map((issue, i) => (
                    <Card key={i} className="p-4 bg-secondary border-l-4 border-l-destructive">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {SEV_ICON[issue.severity] ?? SEV_ICON.Medium}
                          <span className="font-semibold">{issue.title}</span>
                        </div>
                        <Badge variant="outline" className="border-primary/30">
                          {issue.confidence}% confidence
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground space-y-1">
                        <div>Line {issue.line} · {issue.category}</div>
                        <div className="text-xs">Detected by: {issue.detected_by}</div>
                      </div>
                    </Card>
                  ))}
                </div>

                <Card className="p-4 bg-primary/10 border-primary/30">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-primary mt-0.5" />
                    <div>
                      <div className="font-semibold mb-1">ML Insights</div>
                      <p className="text-sm text-muted-foreground">
                        The ML model detected patterns consistent with common security vulnerabilities.
                        Recommended fixes include parameterized queries and environment variables for secrets.
                      </p>
                    </div>
                  </div>
                </Card>

                <div className="flex justify-center pt-2">
                  <Button className="bg-gradient-primary gap-2" onClick={() => navigate("/submit")}>
                    Analyze Your Own Code
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </Card>
        </div>
      </div>
    </section>
  );
};
