import { useNavigate } from "react-router-dom";
import { AppNavigation } from "@/components/app/AppNavigation";
import { StatCard } from "@/components/app/StatCard";
import { QuickActions } from "@/components/app/QuickActions";
import { Button } from "@/components/ui/button";
import { mockStats, mockUser } from "@/data/mockData";
import { getEntries, getDashboardStats } from "@/services/reviewHistory";
import { ExternalLink, Clock, FileCode2, AlertTriangle } from "lucide-react";

const SEV_CLS: Record<string, string> = {
  critical: "bg-red-600 text-white",
  high:     "bg-orange-500 text-white",
  medium:   "bg-yellow-500 text-black",
  low:      "bg-blue-500 text-white",
  none:     "bg-muted text-muted-foreground",
};

const STATUS_CLS: Record<string, string> = {
  completed:   "text-green-400",
  in_progress: "text-yellow-400",
  pending:     "text-blue-400",
};

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

const Dashboard = () => {
  const navigate = useNavigate();
  const realStats = getDashboardStats();
  const recentEntries = getEntries().slice(0, 5);

  const stats = realStats
    ? {
        totalReviews: realStats.totalReviews,
        totalReviewsChange: realStats.thisWeek,
        issuesFound: realStats.issuesFound,
        issuesFoundChange: 0,
        avgReviewTime: "< 30s",
        avgReviewTimeChange: "AI-powered",
        codeQualityScore: realStats.avgScore,
        codeQualityScoreChange: 0,
      }
    : mockStats;

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Welcome Banner */}
        <div className="bg-card border border-border rounded-xl p-6 mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground mb-1">
              Welcome back, {mockUser.name.split(" ")[0]}!
            </h1>
            <p className="text-muted-foreground">
              {recentEntries.length > 0
                ? `You have ${recentEntries.length} review${recentEntries.length !== 1 ? "s" : ""} in your history`
                : "Submit your first code snippet to get started"}
            </p>
          </div>
          <Button className="bg-gradient-primary" onClick={() => navigate("/submit")}>
            + New Analysis
          </Button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Reviews"
            value={stats.totalReviews}
            change={realStats ? `+${stats.totalReviewsChange} this week` : `+${mockStats.totalReviewsChange} this week`}
            changeType="positive"
          />
          <StatCard
            title="Issues Found"
            value={stats.issuesFound.toLocaleString()}
            change={realStats ? "across all analyses" : `+${mockStats.issuesFoundChange} this week`}
            changeType="positive"
          />
          <StatCard
            title="Avg Review Time"
            value={stats.avgReviewTime}
            change={realStats ? "AI-powered instant" : mockStats.avgReviewTimeChange}
            changeType="positive"
          />
          <StatCard
            title="Code Quality Score"
            value={stats.codeQualityScore}
            suffix="/100"
            change={realStats ? "avg across analyses" : `+${mockStats.codeQualityScoreChange} points`}
            changeType="positive"
          />
        </div>

        {/* Recent Reviews */}
        <div className="bg-card border border-border rounded-xl mb-8 overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-border">
            <h2 className="font-semibold text-foreground">Recent Reviews</h2>
            <Button variant="outline" size="sm" onClick={() => navigate("/reviews")}>
              View All
            </Button>
          </div>

          {recentEntries.length === 0 ? (
            <div className="px-6 py-10 text-center text-muted-foreground">
              <FileCode2 className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">No analyses yet. Submit code to see results here.</p>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {recentEntries.map((entry) => (
                <div key={entry.id} className="flex items-center gap-4 px-6 py-4 hover:bg-secondary/10 transition-colors">
                  <FileCode2 className="w-6 h-6 text-primary shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-foreground truncate">{entry.filename}</span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${SEV_CLS[entry.severity] ?? SEV_CLS.none}`}>
                        {entry.severity === "none" ? "Clean" : entry.severity}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <Clock className="w-3 h-3" />
                      <span>{fmt(entry.submittedAt)}</span>
                      <span>·</span>
                      <span>{entry.issueCount} issues</span>
                      <span>·</span>
                      <span>Score {entry.overallScore}/100</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className={`text-xs font-medium capitalize ${STATUS_CLS[entry.status]}`}>
                      {entry.status.replace("_", " ")}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1 text-xs"
                      onClick={() => navigate("/reviews/result", { state: { result: entry.result } })}
                    >
                      <ExternalLink className="w-3.5 h-3.5" /> View
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <QuickActions />
      </main>
    </div>
  );
};

export default Dashboard;
