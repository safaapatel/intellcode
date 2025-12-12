import { Card } from "@/components/ui/card";
import { Brain, GitBranch, TrendingUp, Shield, Zap, Code2 } from "lucide-react";

const features = [
  {
    icon: Code2,
    title: "Static Code Analysis",
    description: "Integrated Pylint, Bandit, and Flake8 for comprehensive code quality checks, security scanning, and style compliance.",
    color: "text-primary",
  },
  {
    icon: Brain,
    title: "ML-Powered Detection",
    description: "CodeBERT model identifies complex issues with context-aware suggestions and intelligent explanations.",
    color: "text-accent",
  },
  {
    icon: GitBranch,
    title: "GitHub Integration",
    description: "Seamless PR integration with automatic analysis, inline comments, and commit status updates.",
    color: "text-primary",
  },
  {
    icon: Shield,
    title: "Security First",
    description: "OWASP-compliant security scanning detects SQL injection, hardcoded secrets, and crypto vulnerabilities.",
    color: "text-accent",
  },
  {
    icon: Zap,
    title: "Real-time Analysis",
    description: "Async processing and caching deliver fast results with WebSocket updates for collaborative reviews.",
    color: "text-primary",
  },
  {
    icon: TrendingUp,
    title: "Analytics Dashboard",
    description: "Track quality trends, team performance, and ROI with interactive charts and AI-generated insights.",
    color: "text-accent",
  },
];

export const Features = () => {
  return (
    <section id="features" className="py-24 relative">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
            Intelligent Features
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Everything you need for world-class code review automation
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, i) => (
            <Card 
              key={i}
              className="p-6 bg-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-glow-primary group"
            >
              <div className="mb-4">
                <feature.icon className={`w-12 h-12 ${feature.color} group-hover:scale-110 transition-transform duration-300`} />
              </div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
