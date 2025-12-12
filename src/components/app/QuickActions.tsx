import { Link } from "react-router-dom";
import { Upload, Settings, BarChart3 } from "lucide-react";

export const QuickActions = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Link
        to="/submit"
        className="flex items-center justify-center gap-3 p-6 bg-card border border-border rounded-xl hover:border-primary/50 hover:bg-secondary/30 transition-all group"
      >
        <Upload className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
        <span className="font-semibold text-foreground group-hover:text-primary transition-colors">Submit Code</span>
      </Link>
      <Link
        to="/rules"
        className="flex items-center justify-center gap-3 p-6 bg-card border border-border rounded-xl hover:border-primary/50 hover:bg-secondary/30 transition-all group"
      >
        <Settings className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
        <span className="font-semibold text-foreground group-hover:text-primary transition-colors">Configure Rules</span>
      </Link>
      <Link
        to="/analytics"
        className="flex items-center justify-center gap-3 p-6 bg-card border border-border rounded-xl hover:border-primary/50 hover:bg-secondary/30 transition-all group"
      >
        <BarChart3 className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
        <span className="font-semibold text-foreground group-hover:text-primary transition-colors">View Analytics</span>
      </Link>
    </div>
  );
};
