import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Github, FileCode } from "lucide-react";
import { mockRepositories, mockBranches, mockPullRequests, mockRuleSets, mockFilesToAnalyze } from "@/data/mockData";
import { toast } from "sonner";

const Submit = () => {
  const navigate = useNavigate();
  const [submissionMethod, setSubmissionMethod] = useState<"upload" | "github">("github");
  const [selectedRepo, setSelectedRepo] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("");
  const [selectedPR, setSelectedPR] = useState("");
  const [selectedRuleSet, setSelectedRuleSet] = useState("");
  const [priority, setPriority] = useState<"normal" | "high" | "critical">("normal");
  const [options, setOptions] = useState({
    securityScan: true,
    codeSmell: true,
    mlSuggestions: true,
    refactoring: true,
  });

  const handleSubmit = () => {
    toast.success("Code submitted for review!", {
      description: "Analysis will complete in approximately 8-12 seconds.",
    });
    setTimeout(() => navigate("/reviews/1"), 1500);
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-3xl">
        {/* Breadcrumb */}
        <div className="text-sm text-muted-foreground mb-4">
          <span className="hover:text-foreground cursor-pointer" onClick={() => navigate("/dashboard")}>Dashboard</span>
          <span className="mx-2">›</span>
          <span className="text-foreground">Submit Code</span>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-8">Submit Code for Review</h1>

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
              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                submissionMethod === "upload" ? "border-primary" : "border-muted-foreground"
              }`}>
                {submissionMethod === "upload" && <div className="w-2 h-2 rounded-full bg-primary" />}
              </div>
              <Upload className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">Upload Files Directly</span>
            </label>
            <label
              className={`flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all ${
                submissionMethod === "github"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground"
              }`}
              onClick={() => setSubmissionMethod("github")}
            >
              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                submissionMethod === "github" ? "border-primary" : "border-muted-foreground"
              }`}>
                {submissionMethod === "github" && <div className="w-2 h-2 rounded-full bg-primary" />}
              </div>
              <Github className="w-5 h-5 text-muted-foreground" />
              <span className="text-foreground">Connect GitHub Repository</span>
            </label>
          </div>
        </div>

        {/* Repository Selection */}
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
                      <SelectItem key={repo.id} value={repo.id}>{repo.fullName}</SelectItem>
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
                      <SelectItem key={branch.name} value={branch.name}>{branch.name}</SelectItem>
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
                      <SelectItem key={pr.id} value={String(pr.id)}>{pr.title}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Configuration */}
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
              <Label className="text-foreground mb-3 block">Analysis Options</Label>
              <div className="space-y-3">
                {[
                  { key: "securityScan", label: "Security scan" },
                  { key: "codeSmell", label: "Code smell detection" },
                  { key: "mlSuggestions", label: "ML-powered suggestions" },
                  { key: "refactoring", label: "Generate refactoring examples" },
                ].map((option) => (
                  <div key={option.key} className="flex items-center gap-3">
                    <Checkbox
                      id={option.key}
                      checked={options[option.key as keyof typeof options]}
                      onCheckedChange={(checked) =>
                        setOptions({ ...options, [option.key]: checked })
                      }
                      className="border-primary data-[state=checked]:bg-primary"
                    />
                    <label htmlFor={option.key} className="text-sm text-foreground cursor-pointer">
                      {option.label}
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Files Preview */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">Files to be analyzed ({mockFilesToAnalyze.length})</h2>
          <div className="space-y-2">
            {mockFilesToAnalyze.map((file, index) => (
              <div key={index} className="flex items-center gap-3 text-sm">
                <FileCode className="w-4 h-4 text-primary" />
                <span className="font-mono text-foreground">{file.path}</span>
                <span className="text-muted-foreground">({file.lines} lines)</span>
              </div>
            ))}
          </div>
          <p className="text-sm text-warning mt-4">
            <span className="font-medium">Estimated analysis time:</span> 8-12 seconds
          </p>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-4">
          <Button variant="outline" onClick={() => navigate("/dashboard")}>Cancel</Button>
          <Button className="bg-gradient-primary" onClick={handleSubmit}>
            Submit for Review
          </Button>
        </div>
      </main>
    </div>
  );
};

export default Submit;
