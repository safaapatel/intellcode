import { Navigate } from "react-router-dom";
import { Hero } from "@/components/Hero";
import { Features } from "@/components/Features";
import { InteractiveDemo } from "@/components/InteractiveDemo";
import { HowItWorks } from "@/components/HowItWorks";
import { CTA } from "@/components/CTA";
import { AppNavigation } from "@/components/app/AppNavigation";
import { getSession } from "@/services/auth";

const Index = () => {
  if (getSession()) return <Navigate to="/dashboard" replace />;

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />
      <Hero />
      <Features />
      <InteractiveDemo />
      <HowItWorks />
      <CTA />
    </div>
  );
};

export default Index;
