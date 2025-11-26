import { Hero } from "@/components/Hero";
import { Features } from "@/components/Features";
import { InteractiveDemo } from "@/components/InteractiveDemo";
import { HowItWorks } from "@/components/HowItWorks";
import { CTA } from "@/components/CTA";
import { Navigation } from "@/components/Navigation";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <Hero />
      <Features />
      <InteractiveDemo />
      <HowItWorks />
      <CTA />
    </div>
  );
};

export default Index;
