import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Play, AlertTriangle, CheckCircle2, XCircle, TrendingUp } from "lucide-react";
import { toast } from "sonner";

const sampleCode = `def process_user_data(user_input):
    # Security issue: SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    
    # Code smell: long method (>50 lines simulated)
    result = execute_query(query)
    
    # Hardcoded secret detected
    api_key = "sk_live_1234567890abcdef"
    
    return result`;

const analysisResults = {
  overall_score: 62,
  issues: [
    {
      severity: "Critical",
      category: "Security",
      title: "SQL Injection Vulnerability",
      line: 3,
      confidence: 98,
      detected_by: "Bandit + ML",
    },
    {
      severity: "High",
      category: "Security",
      title: "Hardcoded Secret Detected",
      line: 9,
      confidence: 95,
      detected_by: "Bandit",
    },
    {
      severity: "Medium",
      category: "Code Smell",
      title: "Function Too Complex",
      line: 1,
      confidence: 87,
      detected_by: "Static Analysis",
    },
  ],
  metrics: {
    nash_threshold: 0.82,
    shapley_contribution: 0.67,
    false_positive_rate: 8,
  },
};

export const InteractiveDemo = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const runAnalysis = () => {
    setIsAnalyzing(true);
    setShowResults(false);
    
    toast.loading("Running static analysis...", { id: "analysis" });
    
    setTimeout(() => {
      toast.loading("ML model analyzing patterns...", { id: "analysis" });
    }, 1000);
    
    setTimeout(() => {
      toast.loading("Calculating game theory metrics...", { id: "analysis" });
    }, 2000);
    
    setTimeout(() => {
      setIsAnalyzing(false);
      setShowResults(true);
      toast.success("Analysis complete!", { id: "analysis" });
    }, 3000);
  };

  return (
    <section className="py-24 bg-secondary/50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-accent bg-clip-text text-transparent">
            See It In Action
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Watch IntelliCode Review analyze real code with all three analysis engines
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <Card className="p-8 bg-card border-primary/20">
            <Tabs defaultValue="code" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="code">Code Sample</TabsTrigger>
                <TabsTrigger value="results" disabled={!showResults}>
                  Analysis Results
                </TabsTrigger>
              </TabsList>

              <TabsContent value="code" className="space-y-6">
                <div className="bg-secondary rounded-lg p-6 font-mono text-sm overflow-x-auto">
                  <pre className="text-foreground">{sampleCode}</pre>
                </div>

                <div className="flex justify-center">
                  <Button
                    size="lg"
                    onClick={runAnalysis}
                    disabled={isAnalyzing}
                    className="bg-gradient-primary hover:shadow-glow-primary"
                  >
                    {isAnalyzing ? (
                      <>Analyzing...</>
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
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-warning mb-1">
                      {analysisResults.overall_score}/100
                    </div>
                    <div className="text-sm text-muted-foreground">Quality Score</div>
                  </Card>
                  
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-primary mb-1">
                      {analysisResults.metrics.nash_threshold}
                    </div>
                    <div className="text-sm text-muted-foreground">Nash Threshold</div>
                  </Card>
                  
                  <Card className="p-4 bg-secondary border-border">
                    <div className="text-3xl font-bold text-accent mb-1">
                      {analysisResults.metrics.shapley_contribution}
                    </div>
                    <div className="text-sm text-muted-foreground">Shapley Value</div>
                  </Card>
                </div>

                <div className="space-y-3">
                  {analysisResults.issues.map((issue, i) => (
                    <Card key={i} className="p-4 bg-secondary border-l-4 border-l-destructive">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {issue.severity === "Critical" ? (
                            <XCircle className="w-5 h-5 text-destructive" />
                          ) : issue.severity === "High" ? (
                            <AlertTriangle className="w-5 h-5 text-warning" />
                          ) : (
                            <TrendingUp className="w-5 h-5 text-primary" />
                          )}
                          <span className="font-semibold">{issue.title}</span>
                        </div>
                        <Badge variant="outline" className="border-primary/30">
                          {issue.confidence}% confidence
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground space-y-1">
                        <div>Line {issue.line} • {issue.category}</div>
                        <div className="text-xs">Detected by: {issue.detected_by}</div>
                      </div>
                    </Card>
                  ))}
                </div>

                <Card className="p-4 bg-primary/10 border-primary/30">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-primary mt-0.5" />
                    <div>
                      <div className="font-semibold mb-1">Game Theory Insights</div>
                      <p className="text-sm text-muted-foreground">
                        Based on Nash equilibrium analysis, the optimal review threshold is 0.82. 
                        This developer's Shapley value of 0.67 indicates strong contribution to code quality.
                      </p>
                    </div>
                  </div>
                </Card>
              </TabsContent>
            </Tabs>
          </Card>
        </div>
      </div>
    </section>
  );
};
