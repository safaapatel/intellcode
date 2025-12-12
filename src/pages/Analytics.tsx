import { useState } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { StatCard } from "@/components/app/StatCard";
import { Button } from "@/components/ui/button";
import { Download, Lightbulb } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from "recharts";
import {
  mockAnalyticsStats,
  mockQualityTrend,
  mockIssueDistribution,
  mockCommonIssues,
  mockReviewerStats,
  mockKeyInsights,
} from "@/data/mockData";

const Analytics = () => {
  const [timeRange, setTimeRange] = useState<"7" | "30" | "90" | "custom">("30");

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold text-foreground">Analytics & Insights</h1>
          <div className="flex items-center gap-2">
            {(["7", "30", "90"] as const).map((range) => (
              <Button
                key={range}
                variant={timeRange === range ? "default" : "outline"}
                size="sm"
                onClick={() => setTimeRange(range)}
              >
                {range} Days
              </Button>
            ))}
            <Button variant="outline" size="sm">Custom</Button>
            <Button size="sm" className="bg-gradient-primary">
              <Download className="w-4 h-4 mr-2" />
              Export Data
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <StatCard
            title="Total Reviews"
            value={mockAnalyticsStats.totalReviews}
            change={mockAnalyticsStats.totalReviewsChange}
            changeType="positive"
          />
          <StatCard
            title="Avg Quality Score"
            value={mockAnalyticsStats.avgQualityScore}
            suffix="/100"
            change={mockAnalyticsStats.avgQualityScoreChange}
            changeType="positive"
          />
          <StatCard
            title="Time Saved"
            value={mockAnalyticsStats.timeSaved}
            change={mockAnalyticsStats.timeSavedDescription}
            changeType="neutral"
          />
        </div>

        {/* Quality Trend Chart */}
        <div className="bg-card border border-border rounded-xl p-6 mb-8">
          <h2 className="text-lg font-semibold text-primary mb-2">Code Quality Trend</h2>
          <p className="text-sm text-muted-foreground mb-6">Average Quality Score Over Time</p>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockQualityTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[60, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="hsl(var(--primary))"
                  strokeWidth={3}
                  dot={{ fill: "hsl(var(--primary))", strokeWidth: 2 }}
                  activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Issue Distribution */}
          <div className="bg-card border border-border rounded-xl p-6">
            <h2 className="text-lg font-semibold text-foreground mb-6">Issue Distribution</h2>
            <div className="h-[250px] flex items-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={mockIssueDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {mockIssueDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 ml-4">
                {mockIssueDistribution.map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-sm text-muted-foreground">{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Most Common Issues */}
          <div className="bg-card border border-border rounded-xl p-6">
            <h2 className="text-lg font-semibold text-foreground mb-6">Most Common Issues</h2>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mockCommonIssues} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={12}
                    width={140}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {mockCommonIssues.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Reviewer Stats Table */}
        <div className="bg-card border border-border rounded-xl overflow-hidden mb-8">
          <div className="px-6 py-4 border-b border-border">
            <h2 className="text-lg font-semibold text-foreground">Reviewer Statistics</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Reviewer</th>
                  <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Reviews</th>
                  <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Avg Time</th>
                  <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Accuracy</th>
                  <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Issues</th>
                </tr>
              </thead>
              <tbody>
                {mockReviewerStats.map((reviewer, index) => (
                  <tr key={index} className="border-b border-border last:border-b-0 hover:bg-secondary/30 transition-colors">
                    <td className="px-6 py-4 font-medium text-primary">{reviewer.name}</td>
                    <td className="px-6 py-4 text-foreground">{reviewer.reviews}</td>
                    <td className="px-6 py-4 text-foreground">{reviewer.avgTime}</td>
                    <td className="px-6 py-4 text-success font-medium">{reviewer.accuracy}</td>
                    <td className="px-6 py-4 text-foreground">{reviewer.issues}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Key Insights */}
        <div className="bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/30 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Lightbulb className="w-5 h-5 text-warning" />
            <h2 className="text-lg font-semibold text-foreground">Key Insights</h2>
          </div>
          <ul className="space-y-2">
            {mockKeyInsights.map((insight, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                <span className="text-primary">✦</span>
                {insight}
              </li>
            ))}
          </ul>
        </div>
      </main>
    </div>
  );
};

export default Analytics;
