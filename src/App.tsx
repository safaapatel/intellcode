import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
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
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/submit" element={<Submit />} />
          <Route path="/reviews/:id" element={<ReviewDetail />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/rules" element={<Rules />} />
          <Route path="/reviews" element={<Reviews />} />
          <Route path="/repositories" element={<Repositories />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/models" element={<Models />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
