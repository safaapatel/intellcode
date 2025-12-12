import { AppNavigation } from "@/components/app/AppNavigation";
import { StatCard } from "@/components/app/StatCard";
import { ReviewsTable } from "@/components/app/ReviewsTable";
import { QuickActions } from "@/components/app/QuickActions";
import { mockStats, mockRecentReviews, mockUser } from "@/data/mockData";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Welcome Banner */}
        <div className="bg-card border border-border rounded-xl p-6 mb-8">
          <h1 className="text-2xl font-bold text-foreground mb-1">Welcome back, {mockUser.name.split(" ")[0]}!</h1>
          <p className="text-muted-foreground">
            You have <span className="text-primary font-medium">3 pending reviews</span> requiring attention
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Reviews"
            value={mockStats.totalReviews}
            change={`+${mockStats.totalReviewsChange} this week`}
            changeType="positive"
          />
          <StatCard
            title="Issues Found"
            value={mockStats.issuesFound.toLocaleString()}
            change={`+${mockStats.issuesFoundChange} this week`}
            changeType="positive"
          />
          <StatCard
            title="Avg Review Time"
            value={mockStats.avgReviewTime}
            change={mockStats.avgReviewTimeChange}
            changeType="positive"
          />
          <StatCard
            title="Code Quality Score"
            value={mockStats.codeQualityScore}
            suffix="/100"
            change={`+${mockStats.codeQualityScoreChange} points`}
            changeType="positive"
          />
        </div>

        {/* Reviews Table */}
        <div className="mb-8">
          <ReviewsTable reviews={mockRecentReviews} />
        </div>

        {/* Quick Actions */}
        <QuickActions />
      </main>
    </div>
  );
};

export default Dashboard;
