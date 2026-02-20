import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Code2 } from "lucide-react";

export const Navigation = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-gradient-primary rounded-lg">
            <Code2 className="w-6 h-6 text-background" />
          </div>
          <span className="text-xl font-bold">IntelliCode Review</span>
        </div>
        
        <div className="hidden md:flex items-center gap-8">
          <Link to="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
            Dashboard
          </Link>
          <Link to="/reviews" className="text-muted-foreground hover:text-foreground transition-colors">
            Reviews
          </Link>
          <Link to="/analytics" className="text-muted-foreground hover:text-foreground transition-colors">
            Analytics
          </Link>
          <Link to="/rules" className="text-muted-foreground hover:text-foreground transition-colors">
            Rules
          </Link>
          <Link to="/settings" className="text-muted-foreground hover:text-foreground transition-colors">
            Settings
          </Link>
        </div>
        
        <div className="flex items-center gap-4">
          <Button variant="ghost" className="hidden sm:inline-flex" asChild>
            <Link to="/login">Sign In</Link>
          </Button>
          <Button className="bg-gradient-primary hover:shadow-glow-primary" asChild>
            <Link to="/login">Get Started</Link>
          </Button>
        </div>
      </div>
    </nav>
  );
};
