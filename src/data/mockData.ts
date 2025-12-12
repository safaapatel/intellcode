// Mock data for the IntelliCode Review app

export const mockUser = {
  id: "1",
  name: "Safaa Patel",
  email: "safaa@example.com",
  avatar: "SP",
  role: "Team Lead",
};

export const mockStats = {
  totalReviews: 147,
  totalReviewsChange: 12,
  issuesFound: 1234,
  issuesFoundChange: 45,
  avgReviewTime: "18 min",
  avgReviewTimeChange: "-5 min faster",
  codeQualityScore: 87,
  codeQualityScoreChange: 3,
};

export const mockRecentReviews = [
  {
    id: "1",
    repository: "frontend/react-app",
    prTitle: "Fix auth bug",
    status: "completed" as const,
    issues: "3 Medium",
    issueCount: 3,
    severity: "medium" as const,
  },
  {
    id: "2",
    repository: "backend/api-service",
    prTitle: "Add validation",
    status: "in_progress" as const,
    issues: "7 High",
    issueCount: 7,
    severity: "high" as const,
  },
  {
    id: "3",
    repository: "ml-models/training",
    prTitle: "Update model",
    status: "pending" as const,
    issues: "-",
    issueCount: 0,
    severity: "none" as const,
  },
];

export const mockRepositories = [
  { id: "1", name: "safaa-patel/code-review", fullName: "safaa-patel/code-review" },
  { id: "2", name: "frontend/react-app", fullName: "frontend/react-app" },
  { id: "3", name: "backend/api-service", fullName: "backend/api-service" },
  { id: "4", name: "ml-models/training", fullName: "ml-models/training" },
];

export const mockBranches = [
  { name: "main" },
  { name: "develop" },
  { name: "feature/add-ml-service" },
  { name: "feature/auth-improvements" },
];

export const mockPullRequests = [
  { id: 42, title: "PR #42: Integrate CodeBERT model" },
  { id: 41, title: "PR #41: Fix authentication flow" },
  { id: 40, title: "PR #40: Add validation middleware" },
];

export const mockRuleSets = [
  {
    id: "1",
    name: "Python Strict",
    description: "Comprehensive Python rules with strict complexity limits",
    language: "Python",
    isDefault: true,
    repositories: 3,
    issues: 156,
  },
  {
    id: "2",
    name: "JavaScript Standard",
    description: "Standard JavaScript/TypeScript linting rules",
    language: "JavaScript",
    isDefault: false,
    repositories: 5,
    issues: 89,
  },
  {
    id: "3",
    name: "Team Backend Default",
    description: "Custom rules for backend services",
    language: "Multiple",
    isDefault: false,
    repositories: 12,
    issues: 234,
  },
  {
    id: "4",
    name: "Minimal Security Only",
    description: "Security-focused rules only",
    language: "Multiple",
    isDefault: false,
    repositories: 7,
    issues: 45,
  },
];

export const mockCodeQualityRules = [
  { id: "1", name: "Max line length", enabled: true, severity: "medium", threshold: 80, unit: "chars" },
  { id: "2", name: "Cyclomatic complexity", enabled: true, severity: "high", threshold: 10, unit: "max" },
  { id: "3", name: "Max function length", enabled: true, severity: "medium", threshold: 50, unit: "lines" },
  { id: "4", name: "Max parameters", enabled: true, severity: "low", threshold: 5, unit: "params" },
  { id: "5", name: "Nesting depth", enabled: true, severity: "medium", threshold: 4, unit: "levels" },
];

export const mockSecurityRules = [
  { id: "1", name: "SQL injection detection", enabled: true, severity: "critical" },
  { id: "2", name: "Hardcoded credentials", enabled: true, severity: "critical" },
  { id: "3", name: "Weak cryptography", enabled: true, severity: "high" },
  { id: "4", name: "Path traversal", enabled: true, severity: "high" },
  { id: "5", name: "XML external entities", enabled: true, severity: "high" },
];

export const mockCodeSmellRules = [
  { id: "1", name: "Duplicate code blocks", enabled: true, severity: "medium", threshold: 6, unit: "lines min" },
  { id: "2", name: "God class detection", enabled: true, severity: "high", threshold: 20, unit: "methods" },
  { id: "3", name: "Dead code detection", enabled: true, severity: "low" },
  { id: "4", name: "Inappropriate intimacy", enabled: true, severity: "medium" },
];

export const mockReviewResult = {
  id: "1",
  overallScore: 83,
  totalIssues: 12,
  criticalCount: 2,
  highCount: 3,
  mediumCount: 5,
  lowCount: 2,
  filesAnalyzed: 4,
  linesAnalyzed: 823,
  duration: "9.4 seconds",
  status: "action_required" as const,
};

export const mockIssues = [
  {
    id: "1",
    severity: "critical" as const,
    category: "Security Vulnerability",
    confidence: 95,
    title: "SQL Injection Risk in User Query",
    filePath: "src/api/user_service.py",
    lineNumbers: "45-47",
    description: "Direct string concatenation in SQL query allows potential injection attacks. User input is not sanitized before database execution.",
    problematicCode: `def get_user(user_id):
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    return db.execute(query)`,
    suggestedFix: `def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))`,
    explanation: "SQL injection vulnerabilities allow attackers to manipulate database queries, potentially exposing sensitive data or destroying records.",
    comments: [],
  },
  {
    id: "2",
    severity: "high" as const,
    category: "Code Smell",
    confidence: 88,
    title: "Complexity: Cyclomatic complexity exceeds threshold",
    filePath: "src/ml_service/inference.py",
    lineNumbers: "120-145",
    description: "Function has cyclomatic complexity of 15, exceeding the threshold of 10. Consider breaking into smaller functions.",
    problematicCode: `def process_data(data, config):
    if data is None:
        return None
    # ... many nested conditions
    for item in data:
        if item.type == 'a':
            if item.status == 'active':
                # complex logic
                pass`,
    suggestedFix: `def process_data(data, config):
    if data is None:
        return None
    return [process_item(item) for item in data]

def process_item(item):
    handlers = {'a': handle_type_a, 'b': handle_type_b}
    return handlers.get(item.type, handle_default)(item)`,
    explanation: "High cyclomatic complexity makes code harder to test, maintain, and understand. Breaking it into smaller functions improves readability.",
    comments: [],
  },
  {
    id: "3",
    severity: "medium" as const,
    category: "Style Violation",
    confidence: 92,
    title: "Line length exceeds 80 characters",
    filePath: "src/api/routes.py",
    lineNumbers: "23, 45, 67",
    description: "Multiple lines exceed the configured maximum line length of 80 characters.",
    problematicCode: `response = make_response(json.dumps({"status": "success", "data": result, "metadata": {"timestamp": datetime.now()}}))`,
    suggestedFix: `response = make_response(json.dumps({
    "status": "success",
    "data": result,
    "metadata": {"timestamp": datetime.now()}
}))`,
    explanation: "Long lines reduce readability and can cause issues in code reviews and diffs.",
    comments: [],
  },
];

export const mockAnalyticsStats = {
  totalReviews: 147,
  totalReviewsChange: "+23% vs prev period",
  avgQualityScore: 83.5,
  avgQualityScoreChange: "+4.2 vs baseline",
  timeSaved: "42.3 hrs",
  timeSavedDescription: "vs manual review",
};

export const mockQualityTrend = [
  { date: "Aug 22", score: 72 },
  { date: "Sep 1", score: 75 },
  { date: "Sep 15", score: 78 },
  { date: "Oct 1", score: 82 },
  { date: "Oct 15", score: 80 },
  { date: "Nov 1", score: 85 },
  { date: "Nov 15", score: 88 },
  { date: "Dec 1", score: 86 },
];

export const mockIssueDistribution = [
  { name: "Low", value: 25, color: "hsl(210, 90%, 60%)" },
  { name: "Medium", value: 35, color: "hsl(38, 92%, 50%)" },
  { name: "High", value: 28, color: "hsl(25, 95%, 53%)" },
  { name: "Critical", value: 12, color: "hsl(0, 85%, 60%)" },
];

export const mockCommonIssues = [
  { name: "Code Complexity", count: 284, color: "hsl(210, 90%, 60%)" },
  { name: "Style Violations", count: 156, color: "hsl(142, 76%, 36%)" },
  { name: "Missing Documentation", count: 89, color: "hsl(38, 92%, 50%)" },
  { name: "Security Vulnerabilities", count: 67, color: "hsl(25, 95%, 53%)" },
  { name: "Code Duplication", count: 45, color: "hsl(0, 85%, 60%)" },
];

export const mockReviewerStats = [
  { name: "Alice Chen", reviews: 42, avgTime: "15 min", accuracy: "94%", issues: 234 },
  { name: "Bob Smith", reviews: 38, avgTime: "22 min", accuracy: "89%", issues: 198 },
  { name: "Carol Davis", reviews: 31, avgTime: "19 min", accuracy: "91%", issues: 167 },
  { name: "Dave Wilson", reviews: 32, avgTime: "25 min", accuracy: "87%", issues: 156 },
];

export const mockKeyInsights = [
  "Code quality improved by 12% this month",
  "Security issues decreased by 34% after new rules",
  "Average review time reduced by 18% with ML suggestions",
  "Alice Chen maintains highest accuracy rate (94%)",
];

export const mockFilesToAnalyze = [
  { path: "src/ml_service/model_loader.py", lines: 234 },
  { path: "src/ml_service/inference.py", lines: 189 },
  { path: "tests/test_ml_service.py", lines: 156 },
  { path: "requirements.txt", lines: 45 },
];
