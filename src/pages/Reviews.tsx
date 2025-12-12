import { useState } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Download, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";

const mockIssues = [
  {
    id: 1,
    severity: "critical",
    category: "Security Vulnerability",
    confidence: 95,
    title: "SQL Injection Risk in User Query",
    file: "src/api/user_service.py",
    lines: "45-47",
    description: "Direct string concatenation in SQL query allows potential injection attacks. User input is not sanitized before database execution.",
    problematicCode: `def get_user(user_id):
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    return db.execute(query)`,
    suggestedFix: `def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))`,
    whyMatters: "SQL injection vulnerabilities allow attackers to manipulate database queries, potentially exposing sensitive data or destroying records.",
    startLine: 45,
  },
  {
    id: 2,
    severity: "high",
    category: "Performance Issue",
    confidence: 88,
    title: "N+1 Query Problem in Loop",
    file: "src/services/order_service.py",
    lines: "112-120",
    description: "Database query executed inside a loop causes N+1 query problem, significantly impacting performance with large datasets.",
    problematicCode: `for order in orders:
    customer = db.query(Customer).filter_by(id=order.customer_id).first()
    order.customer_name = customer.name`,
    suggestedFix: `customer_ids = [order.customer_id for order in orders]
customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()
customer_map = {c.id: c for c in customers}
for order in orders:
    order.customer_name = customer_map[order.customer_id].name`,
    whyMatters: "N+1 queries can cause exponential database load, leading to slow response times and potential service outages under load.",
    startLine: 112,
  },
  {
    id: 3,
    severity: "medium",
    category: "Code Quality",
    confidence: 82,
    title: "Missing Error Handling",
    file: "src/utils/file_handler.py",
    lines: "23-25",
    description: "File operations lack try-catch blocks, which could lead to unhandled exceptions crashing the application.",
    problematicCode: `def read_config(path):
    file = open(path, 'r')
    return json.load(file)`,
    suggestedFix: `def read_config(path):
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read config: {e}")
        return None`,
    whyMatters: "Unhandled exceptions can crash the application and provide poor user experience. Proper error handling ensures graceful degradation.",
    startLine: 23,
  },
  {
    id: 4,
    severity: "low",
    category: "Best Practices",
    confidence: 75,
    title: "Unused Import Statement",
    file: "src/controllers/auth_controller.py",
    lines: "3",
    description: "The 'datetime' module is imported but never used in this file, adding unnecessary overhead.",
    problematicCode: `import datetime  # Unused import`,
    suggestedFix: `# Remove unused import`,
    whyMatters: "Unused imports clutter the codebase and can slightly impact load times. Keeping imports clean improves maintainability.",
    startLine: 3,
  },
  {
    id: 5,
    severity: "critical",
    category: "Security Vulnerability",
    confidence: 92,
    title: "Hardcoded API Key Exposed",
    file: "src/config/settings.py",
    lines: "15",
    description: "API key is hardcoded in source code, which could be exposed if the repository is made public.",
    problematicCode: `API_KEY = "sk_live_abc123xyz789secret"`,
    suggestedFix: `import os
API_KEY = os.environ.get("API_KEY")`,
    whyMatters: "Hardcoded secrets can be easily extracted from source code, leading to unauthorized access and potential data breaches.",
    startLine: 15,
  },
];

const Reviews = () => {
  const [expandedIssue, setExpandedIssue] = useState<number | null>(1);
  const [severityFilter, setSeverityFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  const stats = {
    total: 12,
    critical: 2,
    high: 3,
    medium: 5,
    low: 2,
    score: 83,
    filesAnalyzed: 4,
    linesAnalyzed: 823,
    timeSeconds: 9.4,
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "high":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "medium":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "low":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const filteredIssues = mockIssues.filter((issue) => {
    if (severityFilter !== "all" && issue.severity !== severityFilter) return false;
    if (categoryFilter !== "all" && !issue.category.toLowerCase().includes(categoryFilter)) return false;
    if (searchQuery && !issue.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Score Circle */}
        <div className="flex flex-col items-center mb-8">
          <div className="relative w-28 h-28 mb-4">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="56"
                cy="56"
                r="48"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                className="text-muted/30"
              />
              <circle
                cx="56"
                cy="56"
                r="48"
                stroke="url(#scoreGradient)"
                strokeWidth="8"
                fill="none"
                strokeDasharray={`${(stats.score / 100) * 301.59} 301.59`}
                strokeLinecap="round"
              />
              <defs>
                <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="hsl(var(--primary))" />
                  <stop offset="100%" stopColor="hsl(var(--accent))" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold text-foreground">{stats.score}/100</span>
              <span className="text-xs text-muted-foreground">Code Quality</span>
            </div>
          </div>

          {/* Stats Row */}
          <div className="flex gap-8 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">{stats.total}</div>
              <div className="text-sm text-muted-foreground">Total Issues</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-400">{stats.critical}</div>
              <div className="text-sm text-muted-foreground">Critical</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-400">{stats.high}</div>
              <div className="text-sm text-muted-foreground">High</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">{stats.medium}</div>
              <div className="text-sm text-muted-foreground">Medium</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">{stats.low}</div>
              <div className="text-sm text-muted-foreground">Low</div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-4">
            Analyzed {stats.filesAnalyzed} files, {stats.linesAnalyzed} lines | Completed in {stats.timeSeconds} seconds
          </p>

          {/* Status Banner */}
          <div className="w-full max-w-2xl bg-yellow-500/20 border border-yellow-500/30 rounded-lg py-3 px-6 flex items-center justify-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            <span className="font-medium text-yellow-400">Review Complete - Action Required</span>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-6 items-center">
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-40 bg-card border-border">
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

          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="w-40 bg-card border-border">
              <SelectValue placeholder="All Categories" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="security">Security</SelectItem>
              <SelectItem value="performance">Performance</SelectItem>
              <SelectItem value="quality">Code Quality</SelectItem>
              <SelectItem value="practices">Best Practices</SelectItem>
            </SelectContent>
          </Select>

          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search issues..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-card border-border"
            />
          </div>

          <Button variant="outline" className="gap-2">
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>

        {/* Issues List */}
        <div className="space-y-4">
          {filteredIssues.map((issue) => (
            <div
              key={issue.id}
              className="bg-card border border-border rounded-lg overflow-hidden"
            >
              {/* Issue Header */}
              <div
                className="p-4 cursor-pointer hover:bg-secondary/50 transition-colors"
                onClick={() => setExpandedIssue(expandedIssue === issue.id ? null : issue.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div
                      className={`w-3 h-3 rounded-full mt-1.5 ${
                        issue.severity === "critical"
                          ? "bg-red-500"
                          : issue.severity === "high"
                          ? "bg-orange-500"
                          : issue.severity === "medium"
                          ? "bg-yellow-500"
                          : "bg-blue-500"
                      }`}
                    />
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <Badge className={`${getSeverityColor(issue.severity)} uppercase text-xs`}>
                          {issue.severity}
                        </Badge>
                        <span className="text-sm text-muted-foreground">{issue.category}</span>
                        <span className="text-sm text-muted-foreground">• Confidence: {issue.confidence}%</span>
                      </div>
                      <h3 className="font-semibold text-foreground">{issue.title}</h3>
                      <p className="text-sm text-muted-foreground">
                        File: {issue.file} | Lines {issue.lines}
                      </p>
                    </div>
                  </div>
                  {expandedIssue === issue.id ? (
                    <ChevronUp className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-muted-foreground" />
                  )}
                </div>
              </div>

              {/* Expanded Content */}
              {expandedIssue === issue.id && (
                <div className="px-4 pb-4 border-t border-border pt-4">
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-foreground mb-2">Description:</h4>
                    <p className="text-sm text-muted-foreground">{issue.description}</p>
                  </div>

                  <div className="mb-4">
                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                      Problematic Code
                    </h4>
                    <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 font-mono text-sm overflow-x-auto">
                      {issue.problematicCode.split("\n").map((line, idx) => (
                        <div key={idx} className="flex">
                          <span className="text-muted-foreground w-8 flex-shrink-0">
                            {issue.startLine + idx}
                          </span>
                          <span className="text-red-300">{line}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mb-4">
                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                      Suggested Fix
                    </h4>
                    <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4 font-mono text-sm overflow-x-auto">
                      {issue.suggestedFix.split("\n").map((line, idx) => (
                        <div key={idx} className="flex">
                          <span className="text-muted-foreground w-8 flex-shrink-0">
                            {issue.startLine + idx}
                          </span>
                          <span className="text-green-300">{line}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mb-4 bg-primary/10 border border-primary/20 rounded-lg p-4">
                    <h4 className="font-semibold text-primary mb-1">Why This Matters:</h4>
                    <p className="text-sm text-muted-foreground">{issue.whyMatters}</p>
                  </div>

                  <div className="flex gap-2">
                    <Button size="sm" className="bg-gradient-primary">
                      Mark Resolved
                    </Button>
                    <Button size="sm" variant="outline">
                      Discuss
                    </Button>
                    <Button size="sm" variant="outline">
                      False Positive
                    </Button>
                    <Button size="sm" variant="outline">
                      View in GitHub
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default Reviews;
