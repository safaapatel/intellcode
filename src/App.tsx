import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";
import { ProtectedRoute } from "@/components/app/ProtectedRoute";
import { OnboardingModal } from "@/components/app/OnboardingModal";
import { CommandPalette } from "@/components/app/CommandPalette";
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
        <GitHubOAuthHandler />
        <OnboardingModal />
        <CommandPalette />
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
      </HashRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
