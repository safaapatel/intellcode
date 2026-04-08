import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route } from "react-router-dom";
import { useEffect, Component, type ErrorInfo, type ReactNode } from "react";
import { ProtectedRoute } from "@/components/app/ProtectedRoute";
import { OnboardingModal } from "@/components/app/OnboardingModal";
import { CommandPalette } from "@/components/app/CommandPalette";
import { KeyboardShortcuts } from "@/components/app/KeyboardShortcuts";
import { consumeTokenFromHash, setGitHubToken, getGitHubUser } from "@/services/github";
import { toast } from "sonner";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Submit from "./pages/Submit";
import ReviewDetail from "./pages/ReviewDetail";
import Reviews from "./pages/Reviews";
import Analytics from "./pages/Analytics";
import Rules from "./pages/Rules";
import Repositories from "./pages/Repositories";
import Settings from "./pages/Settings";
import Models from "./pages/Models";
import Admin from "./pages/Admin";
import Compare from "./pages/Compare";
import CI from "./pages/CI";
import Diff from "./pages/Diff";
import Batch from "./pages/Batch";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// ─── Error Boundary ───────────────────────────────────────────────────────────

class AppErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[AppErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-8">
          <div className="max-w-md w-full text-center space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-destructive/10 flex items-center justify-center mx-auto">
              <span className="text-3xl">!</span>
            </div>
            <h1 className="text-xl font-bold text-foreground">Something went wrong</h1>
            <p className="text-sm text-muted-foreground">
              An unexpected error occurred. Your analysis history is safe.
            </p>
            <pre className="text-left text-xs bg-secondary/40 rounded-lg p-3 overflow-x-auto text-muted-foreground">
              {this.state.error.message}
            </pre>
            <button
              type="button"
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium"
              onClick={() => { this.setState({ error: null }); window.location.hash = "/dashboard"; }}
            >
              Go to Dashboard
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/** Reads #github_token=... from URL after OAuth redirect, stores token, shows welcome toast. */
function GitHubOAuthHandler() {
  useEffect(() => {
    const token = consumeTokenFromHash();
    if (!token) return;
    setGitHubToken(token);
    // Fetch user name for a nice toast (best-effort)
    getGitHubUser(true)
      .then((user) => {
        toast.success(`GitHub connected as @${user?.login ?? "you"}`, {
          description: "Your repos are now accessible in Repositories.",
        });
      })
      .catch(() => {
        toast.success("GitHub connected successfully");
      });
  }, []);
  return null;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <HashRouter>
        <AppErrorBoundary>
        <GitHubOAuthHandler />
        <OnboardingModal />
        <CommandPalette />
        <KeyboardShortcuts />
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard"    element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/submit"       element={<ProtectedRoute><Submit /></ProtectedRoute>} />
          <Route path="/reviews/result" element={<ProtectedRoute><ReviewDetail /></ProtectedRoute>} />
          <Route path="/reviews/:id"  element={<ProtectedRoute><ReviewDetail /></ProtectedRoute>} />
          <Route path="/analytics"    element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
          <Route path="/rules"        element={<ProtectedRoute><Rules /></ProtectedRoute>} />
          <Route path="/reviews"      element={<ProtectedRoute><Reviews /></ProtectedRoute>} />
          <Route path="/repositories" element={<ProtectedRoute><Repositories /></ProtectedRoute>} />
          <Route path="/settings"     element={<ProtectedRoute><Settings /></ProtectedRoute>} />
          <Route path="/models"       element={<ProtectedRoute><Models /></ProtectedRoute>} />
          <Route path="/admin"        element={<ProtectedRoute requireRole="admin"><Admin /></ProtectedRoute>} />
          <Route path="/compare"      element={<ProtectedRoute><Compare /></ProtectedRoute>} />
          <Route path="/ci"           element={<ProtectedRoute><CI /></ProtectedRoute>} />
          <Route path="/diff"         element={<ProtectedRoute><Diff /></ProtectedRoute>} />
          <Route path="/batch"        element={<ProtectedRoute><Batch /></ProtectedRoute>} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
        </AppErrorBoundary>
      </HashRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
