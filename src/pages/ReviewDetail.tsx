import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { 
  ChevronDown, 
  ChevronUp, 
  AlertCircle, 
  AlertTriangle, 
  Info, 
  CheckCircle, 
  Download,
  ExternalLink,
  MessageSquare
} from "lucide-react";
import { mockReviewResult, mockIssues } from "@/data/mockData";
import { toast } from "sonner";

const severityConfig = {
  critical: { color: "bg-destructive text-destructive-foreground", icon: AlertCircle, label: "Critical" },
  high: { color: "bg-warning text-background", icon: AlertTriangle, label: "High" },
  medium: { color: "bg-[hsl(38,92%,50%)] text-background", icon: Info, label: "Medium" },
  low: { color: "bg-primary text-primary-foreground", icon: Info, label: "Low" },
};

const ReviewDetail = () => {
  const navigate = useNavigate();
  const [expandedIssue, setExpandedIssue] = useState<string | null>("1");
  const [filters, setFilters] = useState({
    severity: "all",
    category: "all",
    file: "all",
    search: "",
  });

  const getScoreColor = (score: number) => {
    if (score >= 90) return "stroke-success";
    if (score >= 75) return "stroke-primary";
    if (score >= 60) return "stroke-warning";
    return "stroke-destructive";
  };

  const handleMarkResolved = (issueId: string) => {
    toast.success("Issue marked as resolved");
  };

  const handleFalsePositive = (issueId: string) => {
    toast.info("Marked as false positive", { description: "ML model will learn from this feedback" });
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Score Circle & Stats */}
        <div className="bg-card border border-border rounded-xl p-8 mb-6">
          <div className="flex flex-col items-center mb-8">
            <div className="relative w-32 h-32 mb-4">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  fill="none"
                  stroke="hsl(var(--border))"
                  strokeWidth="8"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  fill="none"
                  className={getScoreColor(mockReviewResult.overallScore)}
                  strokeWidth="8"
                  strokeDasharray={`${(mockReviewResult.overallScore / 100) * 352} 352`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold text-foreground">{mockReviewResult.overallScore}</span>
                <span className="text-xs text-muted-foreground">Code Quality</span>
              </div>
            </div>

            {/* Stats Row */}
            <div className="flex items-center justify-center gap-8 text-center">
              <div>
                <p className="text-2xl font-bold text-foreground">{mockReviewResult.totalIssues}</p>
                <p className="text-xs text-muted-foreground">Total Issues</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-destructive">{mockReviewResult.criticalCount}</p>
                <p className="text-xs text-muted-foreground">Critical</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-warning">{mockReviewResult.highCount}</p>
                <p className="text-xs text-muted-foreground">High</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-[hsl(38,92%,50%)]">{mockReviewResult.mediumCount}</p>
                <p className="text-xs text-muted-foreground">Medium</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">{mockReviewResult.lowCount}</p>
                <p className="text-xs text-muted-foreground">Low</p>
              </div>
            </div>

            <p className="text-sm text-muted-foreground mt-4">
              Analyzed {mockReviewResult.filesAnalyzed} files, {mockReviewResult.linesAnalyzed} lines | Completed in {mockReviewResult.duration}
            </p>
          </div>

          {/* Status Banner */}
          <div className="bg-warning/10 border border-warning/30 rounded-lg p-4 flex items-center justify-center gap-2">
            <AlertTriangle className="w-5 h-5 text-warning" />
            <span className="text-warning font-medium">Review Complete - Action Required</span>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-card border border-border rounded-xl p-4 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <Select value={filters.severity} onValueChange={(v) => setFilters({ ...filters, severity: v })}>
              <SelectTrigger className="w-[140px] bg-input">
                <SelectValue placeholder="All Severities" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filters.category} onValueChange={(v) => setFilters({ ...filters, category: v })}>
              <SelectTrigger className="w-[140px] bg-input">
                <SelectValue placeholder="All Categories" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                <SelectItem value="security">Security</SelectItem>
                <SelectItem value="code_smell">Code Smell</SelectItem>
                <SelectItem value="style">Style</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filters.file} onValueChange={(v) => setFilters({ ...filters, file: v })}>
              <SelectTrigger className="w-[140px] bg-input">
                <SelectValue placeholder="All Files" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Files</SelectItem>
              </SelectContent>
            </Select>
            <Input
              placeholder="Search issues..."
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
              className="flex-1 min-w-[200px] bg-input"
            />
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
          </div>
        </div>

        {/* Issues List */}
        <div className="space-y-4">
          {mockIssues.map((issue) => {
            const config = severityConfig[issue.severity];
            const Icon = config.icon;
            const isExpanded = expandedIssue === issue.id;

            return (
              <div key={issue.id} className="bg-card border border-border rounded-xl overflow-hidden">
                {/* Issue Header */}
                <button
                  className="w-full flex items-start gap-4 p-4 text-left hover:bg-secondary/30 transition-colors"
                  onClick={() => setExpandedIssue(isExpanded ? null : issue.id)}
                >
                  <div className={`p-1 rounded-full ${config.color}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${config.color}`}>
                        {config.label}
                      </span>
                      <span className="text-xs text-muted-foreground">{issue.category}</span>
                      <span className="text-xs text-muted-foreground">• Confidence: {issue.confidence}%</span>
                    </div>
                    <h3 className="font-semibold text-foreground">{issue.title}</h3>
                    <p className="text-sm text-muted-foreground font-mono">
                      File: {issue.filePath} | Lines {issue.lineNumbers}
                    </p>
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-muted-foreground" />
                  )}
                </button>

                {/* Expanded Content */}
                {isExpanded && (
                  <div className="px-4 pb-4 space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">Description:</h4>
                      <p className="text-sm text-foreground">{issue.description}</p>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-destructive mb-2">PROBLEMATIC CODE</h4>
                      <pre className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 overflow-x-auto">
                        <code className="text-sm font-mono text-foreground">{issue.problematicCode}</code>
                      </pre>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-success mb-2">SUGGESTED FIX</h4>
                      <pre className="bg-success/10 border border-success/30 rounded-lg p-4 overflow-x-auto">
                        <code className="text-sm font-mono text-foreground">{issue.suggestedFix}</code>
                      </pre>
                    </div>

                    <div className="bg-primary/5 border-l-4 border-primary rounded-r-lg p-4">
                      <h4 className="text-sm font-semibold text-foreground mb-2">Why This Matters:</h4>
                      <p className="text-sm text-muted-foreground">{issue.explanation}</p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-2">
                      <Button size="sm" onClick={() => handleMarkResolved(issue.id)}>
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Mark Resolved
                      </Button>
                      <Button variant="outline" size="sm">
                        <MessageSquare className="w-4 h-4 mr-1" />
                        Discuss
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => handleFalsePositive(issue.id)}>
                        False Positive
                      </Button>
                      <Button variant="outline" size="sm">
                        <ExternalLink className="w-4 h-4 mr-1" />
                        View in GitHub
                      </Button>
                    </div>

                    <div className="text-sm text-muted-foreground">0 comments</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Bottom Actions */}
        <div className="flex items-center justify-between mt-8">
          <Button variant="outline" onClick={() => navigate("/dashboard")}>
            ← Back to Dashboard
          </Button>
          <div className="flex items-center gap-3">
            <Button variant="outline" className="border-warning text-warning hover:bg-warning/10">
              Request Changes
            </Button>
            <Button className="bg-success hover:bg-success/90 text-background">
              Approve PR
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ReviewDetail;
