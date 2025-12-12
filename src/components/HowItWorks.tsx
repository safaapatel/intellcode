import { Card } from "@/components/ui/card";
import { ArrowRight } from "lucide-react";

const steps = [
  {
    number: "01",
    title: "Submit Code",
    description: "Connect your GitHub repo or upload files directly. Configure analysis priority and rule sets.",
  },
  {
    number: "02",
    title: "Multi-Engine Analysis",
    description: "Static analyzers, ML models, and code smell detectors run in parallel for comprehensive review.",
  },
  {
    number: "03",
    title: "Smart Issue Detection",
    description: "Advanced ML models identify complex patterns and provide context-aware suggestions.",
  },
  {
    number: "04",
    title: "Get Actionable Insights",
    description: "Receive prioritized issues with explanations, suggestions, and automated GitHub PR comments.",
  },
];

export const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/30 to-background" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
            How It Works
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Four simple steps to revolutionize your code review process
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-6">
          {steps.map((step, i) => (
            <div key={i} className="flex gap-4 items-start">
              <Card className="p-6 flex-1 bg-card border-border hover:border-primary/50 transition-all duration-300 group">
                <div className="flex items-start gap-6">
                  <div className="text-6xl font-bold text-primary/20 group-hover:text-primary/40 transition-colors">
                    {step.number}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-2xl font-semibold mb-3">{step.title}</h3>
                    <p className="text-muted-foreground leading-relaxed">{step.description}</p>
                  </div>
                </div>
              </Card>
              
              {i < steps.length - 1 && (
                <ArrowRight className="w-6 h-6 text-primary mt-8 hidden lg:block" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
