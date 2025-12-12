import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

interface Review {
  id: string;
  repository: string;
  prTitle: string;
  status: "completed" | "in_progress" | "pending";
  issues: string;
  severity: "critical" | "high" | "medium" | "low" | "none";
}

interface ReviewsTableProps {
  reviews: Review[];
}

export const ReviewsTable = ({ reviews }: ReviewsTableProps) => {
  const getStatusBadge = (status: Review["status"]) => {
    const styles = {
      completed: "bg-success/20 text-success",
      in_progress: "bg-warning/20 text-warning",
      pending: "bg-muted text-muted-foreground",
    };
    const labels = {
      completed: "Completed",
      in_progress: "In Progress",
      pending: "Pending",
    };
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium ${styles[status]}`}>
        {labels[status]}
      </span>
    );
  };

  const getActionButton = (status: Review["status"], id: string) => {
    if (status === "pending") {
      return (
        <Button size="sm" className="bg-success hover:bg-success/90 text-background">
          Start
        </Button>
      );
    }
    return (
      <Button size="sm" asChild>
        <Link to={`/reviews/${id}`}>View</Link>
      </Button>
    );
  };

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-border">
        <h3 className="text-lg font-semibold text-foreground">Recent Reviews</h3>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Repository</th>
              <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">PR Title</th>
              <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Status</th>
              <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Issues</th>
              <th className="text-left text-sm font-medium text-muted-foreground px-6 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {reviews.map((review) => (
              <tr key={review.id} className="border-b border-border last:border-b-0 hover:bg-secondary/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-mono text-sm text-primary">{review.repository}</span>
                </td>
                <td className="px-6 py-4 text-sm text-foreground">{review.prTitle}</td>
                <td className="px-6 py-4">{getStatusBadge(review.status)}</td>
                <td className="px-6 py-4 text-sm text-muted-foreground">{review.issues}</td>
                <td className="px-6 py-4">{getActionButton(review.status, review.id)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
