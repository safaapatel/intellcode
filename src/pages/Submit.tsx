import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Github, Loader2, AlertCircle, FolderOpen } from "lucide-react";
import { useRef } from "react";
import { mockRepositories, mockBranches, mockPullRequests, mockRuleSets } from "@/data/mockData";
import { toast } from "sonner";
import { analyzeCode } from "@/services/api";

const SAMPLE_CODE = `import os
import json
from typing import List

def process_user_data(users, db_connection):
    results = []
    for user in users:
        query = "SELECT * FROM users WHERE id = " + str(user['id'])
        data = db_connection.execute(query)

        # Calculate metrics
        total = 0
        for i in range(len(data)):
            for j in range(len(data)):
                total += data[i]['value'] * data[j]['weight']

        results.append(total)
    return results

def validate_input(data):
    if data != None:
        if data['type'] == 'A':
            if data['value'] > 0:
                if data['enabled'] == True:
                    return True
    return False

secret_key = "hardcoded-secret-key-12345"
api_token = "sk-prod-abc123xyz"
`;

const Submit = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [submissionMethod, setSubmissionMethod] = useState<"upload" | "github">("upload");
  const [selectedRepo, setSelectedRepo] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("");
  const [selectedPR, setSelectedPR] = useState("");
  const [selectedRuleSet, setSelectedRuleSet] = useState("");
  const [priority, setPriority] = useState<"normal" | "high" | "critical">("normal");
  const [code, setCode] = useState(SAMPLE_CODE);
  const [filename, setFilename] = useState("snippet.py");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState({
    securityScan: true,
    codeSmell: true,
    mlSuggestions: true,
    refactoring: true,
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setCode(text);
      setFilename(file.name);
      toast.success(`Loaded ${file.name} (${text.split("\n").length} lines)`);
    };
    reader.readAsText(file);
  };

  const handleSubmit = async () => {
    if (!code.trim()) {
      toast.error("Please paste some code to analyze.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    toast.info("Running all 12 ML models...", {
      description: "Security, complexity, bugs, clones, debt, docs, performance & more.",
    });

    try {
      const result = await analyzeCode(code, filename, "python");
      toast.success("Analysis complete!", { description: result.summary });
      navigate("/reviews/result", { state: { result } });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(
        `Backend error: ${msg}. Make sure the Python backend is running on localhost:8000.`
      );
      toast.error("Analysis failed", { description: msg });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-3xl">
        {/* Breadcrumb */}
        <div className="text-sm text-muted-foreground mb-4">
          <span
            className="hover:text-foreground cursor-pointer"
            onClick={() => navigate("/dashboard")}
          >
            Dashboard
          </span>
          <span className="mx-2">›</span>
          <span className="text-foreground">Submit Code</span>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-2">Submit Code for Review</h1>
        <p className="text-muted-foreground mb-8">12 ML models run in parallel on your code.</p>

        {/* Submission Method */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">Submission Method</h2>
          <div className="space-y-3">
            <label
              className={`flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all ${
                submissionMethod === "upload"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground"
              }`}
              onClick={() => setSubmissionMethod("upload")}
            >
              <div
                className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                  submissionMethod === "upload" ? "border-primary" : "border-muted-foreground"
                }`}
              >
                {submissionMethod === "upload" && (
                  <div className="w-2 h-2 rounded-full bg-primary" />
                )}
              </div>
              <Upload className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">Paste Code Directly</span>
            </label>
            <label
              className={`flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all ${
                submissionMethod === "github"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground"
              }`}
              onClick={() => setSubmissionMethod("github")}
            >
              <div
                className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                  submissionMethod === "github" ? "border-primary" : "border-muted-foreground"
                }`}
              >
                {submissionMethod === "github" && (
                  <div className="w-2 h-2 rounded-full bg-primary" />
                )}
              </div>
              <Github className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">Connect GitHub Repository</span>
            </label>
          </div>
        </div>

        {/* Code Editor */}
        {submissionMethod === "upload" && (
          <div className="bg-card border border-border rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-primary">Code Editor</h2>
              <Button variant="outline" size="sm" className="gap-1.5" onClick={() => fileInputRef.current?.click()}>
                <FolderOpen className="w-4 h-4" /> Open File
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".py,.js,.ts,.java,.txt"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>
            <div className="mb-4">
              <Label className="text-foreground mb-2 block">Filename</Label>
              <Input
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                placeholder="snippet.py"
                className="bg-input border-border font-mono"
              />
            </div>
            <div>
              <Label className="text-foreground mb-2 block">
                Python Code{" "}
                <span className="text-muted-foreground font-normal">(edit or paste below)</span>
              </Label>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                rows={22}
                spellCheck={false}
                className="w-full rounded-lg bg-[hsl(220,26%,4%)] border border-border font-mono text-sm text-foreground p-4 resize-y focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="Paste your Python code here..."
              />
              <p className="text-xs text-muted-foreground mt-2">
                {code.split("\n").length} lines · {code.length} chars
              </p>
            </div>
          </div>
        )}

        {/* GitHub (coming soon) */}
        {submissionMethod === "github" && (
          <div className="bg-card border border-border rounded-xl p-6 mb-6">
            <h2 className="text-lg font-semibold text-primary mb-4">Repository Selection</h2>
            <div className="space-y-4">
              <div>
                <Label className="text-foreground mb-2 block">Repository</Label>
                <Select value={selectedRepo} onValueChange={setSelectedRepo}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue placeholder="Select repository" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockRepositories.map((repo) => (
                      <SelectItem key={repo.id} value={repo.id}>
                        {repo.fullName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-foreground mb-2 block">Branch</Label>
                <Select value={selectedBranch} onValueChange={setSelectedBranch}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue placeholder="Select branch" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockBranches.map((branch) => (
                      <SelectItem key={branch.name} value={branch.name}>
                        {branch.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-foreground mb-2 block">Pull Request</Label>
                <Select value={selectedPR} onValueChange={setSelectedPR}>
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue placeholder="Select pull request (optional)" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockPullRequests.map((pr) => (
                      <SelectItem key={pr.id} value={String(pr.id)}>
                        {pr.title}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-sm text-muted-foreground">
                GitHub integration coming soon. Switch to "Paste Code Directly" to analyze now.
              </p>
            </div>
          </div>
        )}

        {/* Configuration */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-primary mb-4">Analysis Configuration</h2>
          <div className="space-y-4">
            <div>
              <Label className="text-foreground mb-2 block">Rule Set</Label>
              <Select value={selectedRuleSet} onValueChange={setSelectedRuleSet}>
                <SelectTrigger className="bg-input border-border">
                  <SelectValue placeholder="Select rule set" />
                </SelectTrigger>
                <SelectContent>
                  {mockRuleSets.map((ruleSet) => (
                    <SelectItem key={ruleSet.id} value={ruleSet.id}>
                      {ruleSet.name} ({ruleSet.language})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-foreground mb-3 block">Priority</Label>
              <div className="grid grid-cols-3 gap-2">
                {(["normal", "high", "critical"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPriority(p)}
                    className={`py-2 px-4 rounded-lg text-sm font-medium transition-all capitalize ${
                      priority === p
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-foreground hover:bg-secondary/80"
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <Label className="text-foreground mb-3 block">Focus Areas</Label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { key: "securityScan", label: "Security detection" },
                  { key: "codeSmell", label: "Code smells & patterns" },
                  { key: "mlSuggestions", label: "Bug & complexity" },
                  { key: "refactoring", label: "Refactoring suggestions" },
                ].map((opt) => (
                  <div key={opt.key} className="flex items-center gap-3">
                    <Checkbox
                      id={opt.key}
                      checked={options[opt.key as keyof typeof options]}
                      onCheckedChange={(checked) =>
                        setOptions({ ...options, [opt.key]: !!checked })
                      }
                      className="border-primary data-[state=checked]:bg-primary"
                    />
                    <label htmlFor={opt.key} className="text-sm text-foreground cursor-pointer">
                      {opt.label}
                    </label>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                All 12 ML models always run — clones, debt, docs, performance & readability
                included.
              </p>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-destructive/10 border border-destructive/30 rounded-xl p-4 mb-6 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-4">
          <Button
            variant="outline"
            onClick={() => navigate("/dashboard")}
            disabled={isAnalyzing}
          >
            Cancel
          </Button>
          <Button
            className="bg-gradient-primary min-w-[180px]"
            onClick={handleSubmit}
            disabled={isAnalyzing || submissionMethod === "github"}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Run All 12 Models →"
            )}
          </Button>
        </div>
      </main>
    </div>
  );
};

export default Submit;
