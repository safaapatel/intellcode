import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, Github } from "lucide-react";

export const CTA = () => {
  return (
    <section className="py-24 relative">
      <div className="absolute inset-0 bg-gradient-hero" />

      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
            Ready to Transform Your Code Reviews?
          </h2>

          <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto">
            Join thousands of developers using AI and machine learning to ship better code faster
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button
              size="lg"
              className="bg-gradient-primary hover:shadow-glow-primary transition-all duration-300 text-lg px-8 py-6"
              asChild
            >
              <Link to="/login">
                Get Started Free
                <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
            </Button>

            <Button
              size="lg"
              variant="outline"
              className="border-primary/30 hover:border-primary hover:bg-secondary text-lg px-8 py-6"
              onClick={() => window.open("https://github.com/safaapatel/intellcode", "_blank")}
            >
              <Github className="mr-2 w-5 h-5" />
              View on GitHub
            </Button>
          </div>

          <div className="mt-12 flex items-center justify-center gap-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-success" />
              Free & open source
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-success" />
              No credit card required
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-success" />
              12 ML models included
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
