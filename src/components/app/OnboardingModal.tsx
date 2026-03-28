import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Code2, GitBranch, BarChart2, Shield, X, ChevronRight } from "lucide-react";
import { getSession } from "@/services/auth";

const KEY = "intellcode_onboarded";

const STEPS = [
  {
    icon: Code2,
    title: "Submit Code for Analysis",
    description:
      "Paste any Python snippet or file into the Submit page and run it through all ML models at once. Results appear in under 2 seconds.",
    action: "Go to Submit",
    path: "/submit",
  },
  {
    icon: Shield,
    title: "Explore Your Report",
    description:
      "The Review Detail page shows 14 tabs: security findings, complexity metrics, bug predictions, refactoring tips, a Fix Guide with copy-paste code, and more.",
    action: "View a sample",
    path: "/reviews",
  },
  {
    icon: GitBranch,
    title: "Connect a Repository",
    description:
      "Link a GitHub repo and run a batch scan across all its Python files. IntelliCode Review fetches up to 20 source files and produces a per-file quality breakdown.",
    action: "Connect repo",
    path: "/repositories",
  },
  {
    icon: BarChart2,
    title: "Track Quality Over Time",
    description:
      "Submit the same file multiple times and watch the Analytics page build per-file quality trajectories. See exactly which improvements moved the score.",
    action: "Open Analytics",
    path: "/analytics",
  },
];

export function OnboardingModal() {
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    // Only show after login, once per account
    const session = getSession();
    if (!session) return;
    const seen = localStorage.getItem(KEY);
    if (!seen) setVisible(true);
  }, []);

  const dismiss = () => {
    localStorage.setItem(KEY, "1");
    setVisible(false);
  };

  const handleAction = () => {
    localStorage.setItem(KEY, "1");
    setVisible(false);
    navigate(STEPS[step].path);
  };

  const next = () => {
    if (step < STEPS.length - 1) {
      setStep((s) => s + 1);
    } else {
      dismiss();
    }
  };

  if (!visible) return null;

  const current = STEPS[step];
  const Icon = current.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={dismiss} />

      {/* Modal */}
      <div className="relative bg-card border border-border rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
        {/* Progress bar */}
        <div className="h-1 bg-secondary">
          <div
            className="h-full bg-gradient-primary transition-all duration-500"
            style={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
          />
        </div>

        {/* Close */}
        <button
          onClick={dismiss}
          className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Content */}
        <div className="p-8">
          {/* Step pills */}
          <div className="flex gap-1.5 mb-6">
            {STEPS.map((_, i) => (
              <div
                key={i}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  i <= step ? "bg-primary" : "bg-border"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/15 mb-5">
            <Icon className="w-7 h-7 text-primary" />
          </div>

          <p className="text-xs text-primary font-semibold uppercase tracking-wider mb-2">
            Step {step + 1} of {STEPS.length}
          </p>
          <h2 className="text-xl font-bold text-foreground mb-3">{current.title}</h2>
          <p className="text-sm text-muted-foreground leading-relaxed mb-8">
            {current.description}
          </p>

          <div className="flex gap-3">
            <Button
              variant="outline"
              className="flex-1"
              onClick={next}
            >
              {step < STEPS.length - 1 ? "Skip" : "Done"}
            </Button>
            <Button
              className="flex-1 bg-gradient-primary"
              onClick={handleAction}
            >
              {current.action}
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
