import { Link } from "react-router-dom";
import { Code2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getSession } from "@/services/auth";

const NotFound = () => {
  const session = getSession();
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-6 px-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="p-1.5 bg-gradient-primary rounded-lg">
          <Code2 className="w-5 h-5 text-background" />
        </div>
        <span className="text-lg font-bold text-foreground">IntelliCode Review</span>
      </div>
      <div className="text-center">
        <h1 className="text-8xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">404</h1>
        <p className="text-xl font-semibold text-foreground mb-2">Page not found</p>
        <p className="text-muted-foreground mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Button className="bg-gradient-primary gap-2" asChild>
          <Link to={session ? "/dashboard" : "/"}>
            <ArrowLeft className="w-4 h-4" />
            {session ? "Back to Dashboard" : "Back to Home"}
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
