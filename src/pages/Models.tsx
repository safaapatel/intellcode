import { useState } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Shield,
  BarChart2,
  Bug,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  FlaskConical,
  TrendingUp,
  Layers,
  GitBranch,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface ProductionModel {
  id: string;
  name: string;
  status: "production" | "development" | "planned";
  icon: typeof Brain;
  iconColor: string;
  iconBg: string;
  architecture: string;
  trainingData: string;
  useCase: string;
  techStack: string[];
  metrics: { label: string; value: string; pct?: number }[];
  description: string;
}

interface GoalModel {
  id: string;
  name: string;
  goal: string;
  icon: typeof Bug;
  iconColor: string;
  iconBg: string;
  approach: {
    title: string;
    items: string[];
  }[];
  expectedAccuracy: string;
  progress: number;
  progressLabel: string;
}

const productionModels: ProductionModel[] = [
  {
    id: "pattern",
    name: "Pattern Recognition Model",
    status: "production",
    icon: Brain,
    iconColor: "text-primary",
    iconBg: "bg-primary/10",
    architecture: "Fine-tuned CodeBERT transformer model",
    trainingData: "50k+ open-source repositories from GitHub",
    useCase: "Identifies code patterns, anti-patterns, and code smells",
    techStack: ["PyTorch", "Hugging Face Transformers", "Python"],
    metrics: [
      { label: "Benchmark Accuracy", value: "87%", pct: 87 },
      { label: "Training Repos", value: "50k+", pct: 100 },
      { label: "Avg Inference Time", value: "1.2s", pct: 88 },
    ],
    description:
      "Built on CodeBERT — a bimodal pre-trained model for programming language and natural language — the Pattern Recognition Model is fine-tuned to detect anti-patterns, code smells, and stylistic violations across multiple languages.",
  },
  {
    id: "security",
    name: "Security Vulnerability Detection",
    status: "production",
    icon: Shield,
    iconColor: "text-red-400",
    iconBg: "bg-red-500/10",
    architecture: "Ensemble model (Random Forest + CNN)",
    trainingData: "AST patterns, token sequences, control flow graphs",
    useCase: "Detects SQL injection, XSS, hardcoded credentials, etc.",
    techStack: ["scikit-learn", "TensorFlow", "Python"],
    metrics: [
      { label: "Precision", value: "92%", pct: 92 },
      { label: "Recall", value: "85%", pct: 85 },
      { label: "F1 Score", value: "88%", pct: 88 },
    ],
    description:
      "An ensemble of Random Forest and Convolutional Neural Network models trained on abstract syntax trees (ASTs) and control flow graphs. It detects OWASP Top 10 vulnerabilities including SQL injection, XSS, hardcoded credentials, path traversal, and weak cryptography.",
  },
  {
    id: "complexity",
    name: "Code Complexity Prediction",
    status: "production",
    icon: BarChart2,
    iconColor: "text-yellow-400",
    iconBg: "bg-yellow-500/10",
    architecture: "Gradient Boosted Trees (XGBoost)",
    trainingData: "Cyclomatic complexity, nesting depth, line count, unique operators",
    useCase: "Predicts maintainability issues",
    techStack: ["XGBoost", "NumPy", "Python"],
    metrics: [
      { label: "R² Score", value: "0.89", pct: 89 },
      { label: "Feature Coverage", value: "12 metrics", pct: 75 },
      { label: "Prediction Speed", value: "<50ms", pct: 95 },
    ],
    description:
      "A gradient-boosted tree model that predicts code maintainability from static features including cyclomatic complexity, cognitive complexity, nesting depth, Halstead metrics, and operator/operand counts. Achieves R² = 0.89 on held-out benchmarks.",
  },
];

const goalModels: GoalModel[] = [
  {
    id: "bug-prediction",
    name: "Bug Prediction Model",
    goal: "Predict the likelihood of bugs in code before they happen — flagging high-risk files and functions during PR review.",
    icon: Bug,
    iconColor: "text-orange-400",
    iconBg: "bg-orange-500/10",
    approach: [
      {
        title: "Data Collection",
        items: [
          "Mine historical bug data from open-source GitHub projects",
          "Extract commit history, issue tracker links, and code churn metrics",
          "Label files/functions with bug-inducing commit annotations",
        ],
      },
      {
        title: "Feature Engineering",
        items: [
          "Code churn (lines added/deleted over time)",
          "Author experience and contribution history",
          "File age, ownership entropy",
          "Complexity metrics from existing models",
        ],
      },
      {
        title: "Model Architecture",
        items: [
          "Baseline: Logistic Regression for interpretability",
          "Final: Neural Network ensemble for improved accuracy",
          "Class balancing via SMOTE (bug instances are rare)",
        ],
      },
    ],
    expectedAccuracy: "75–80% precision",
    progress: 20,
    progressLabel: "Dataset collection in progress",
  },
];

const statusConfig = {
  production: { label: "Production", color: "bg-green-500/20 text-green-400 border-green-500/30" },
  development: { label: "In Development", color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" },
  planned: { label: "Planned", color: "bg-muted text-muted-foreground border-border" },
};

const Models = () => {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Cpu className="w-5 h-5 text-primary" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">ML Models</h1>
          </div>
          <p className="text-muted-foreground">
            IntelliCode Review is powered by a suite of machine learning models trained on millions of lines of open-source code.
          </p>
        </div>

        {/* Summary bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          {[
            { icon: CheckCircle2, color: "text-green-400", label: "Production Models", value: "3" },
            { icon: FlaskConical, color: "text-yellow-400", label: "In Development", value: "1" },
            { icon: Database, color: "text-primary", label: "Training Repos", value: "50k+" },
            { icon: TrendingUp, color: "text-orange-400", label: "Avg Accuracy", value: "88%" },
          ].map(({ icon: Icon, color, label, value }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-4 flex items-center gap-3">
              <div className="p-2 bg-secondary rounded-lg flex-shrink-0">
                <Icon className={`w-4 h-4 ${color}`} />
              </div>
              <div>
                <div className="text-xl font-bold text-foreground">{value}</div>
                <div className="text-xs text-muted-foreground">{label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Production Models */}
        <section className="mb-10">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 className="w-4 h-4 text-green-400" />
            <h2 className="text-lg font-semibold text-foreground">Production Models</h2>
            <Badge className="bg-green-500/20 text-green-400 border-green-500/30 ml-1">3 Active</Badge>
          </div>

          <div className="space-y-4">
            {productionModels.map((model) => {
              const { label, color } = statusConfig[model.status];
              const Icon = model.icon;
              const isExpanded = expandedModel === model.id;

              return (
                <div key={model.id} className="bg-card border border-border rounded-xl overflow-hidden">
                  {/* Card Header */}
                  <div
                    className="p-5 cursor-pointer hover:bg-secondary/30 transition-colors"
                    onClick={() => setExpandedModel(isExpanded ? null : model.id)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-4">
                        <div className={`p-2.5 ${model.iconBg} rounded-xl flex-shrink-0`}>
                          <Icon className={`w-5 h-5 ${model.iconColor}`} />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-1">
                            <h3 className="font-semibold text-foreground">{model.name}</h3>
                            <Badge className={`${color} text-xs`}>{label}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">{model.useCase}</p>
                          <div className="flex flex-wrap gap-1.5">
                            {model.techStack.map((t) => (
                              <span key={t} className="px-2 py-0.5 bg-secondary text-xs text-muted-foreground rounded-md border border-border">
                                {t}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-6 flex-shrink-0">
                        {/* Quick metrics */}
                        <div className="hidden md:flex gap-5">
                          {model.metrics.slice(0, 2).map((m) => (
                            <div key={m.label} className="text-center">
                              <div className="text-lg font-bold text-foreground">{m.value}</div>
                              <div className="text-xs text-muted-foreground whitespace-nowrap">{m.label}</div>
                            </div>
                          ))}
                        </div>
                        {isExpanded ? (
                          <ChevronUp className="w-5 h-5 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {isExpanded && (
                    <div className="border-t border-border px-5 pb-5 pt-4">
                      <p className="text-sm text-muted-foreground mb-5">{model.description}</p>

                      <div className="grid md:grid-cols-2 gap-6">
                        {/* Left: Architecture info */}
                        <div className="space-y-4">
                          <div>
                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                              <Layers className="w-3.5 h-3.5" />
                              Architecture
                            </div>
                            <p className="text-sm text-foreground bg-secondary/30 rounded-lg px-3 py-2 border border-border">
                              {model.architecture}
                            </p>
                          </div>
                          <div>
                            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                              <Database className="w-3.5 h-3.5" />
                              Training Data
                            </div>
                            <p className="text-sm text-foreground bg-secondary/30 rounded-lg px-3 py-2 border border-border">
                              {model.trainingData}
                            </p>
                          </div>
                        </div>

                        {/* Right: Metrics bars */}
                        <div>
                          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                            Performance Metrics
                          </div>
                          <div className="space-y-3">
                            {model.metrics.map((m) => (
                              <div key={m.label}>
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-sm text-foreground">{m.label}</span>
                                  <span className="text-sm font-semibold text-foreground">{m.value}</span>
                                </div>
                                {m.pct !== undefined && (
                                  <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-gradient-to-r from-primary to-primary/60 transition-all"
                                      style={{ width: `${m.pct}%` }}
                                    />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Short-Term Goals */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-4 h-4 text-yellow-400" />
            <h2 className="text-lg font-semibold text-foreground">Short-Term ML Goals</h2>
            <span className="text-sm text-muted-foreground ml-1">Next 1–2 months</span>
          </div>

          <div className="space-y-4">
            {goalModels.map((model) => {
              const Icon = model.icon;
              return (
                <div key={model.id} className="bg-card border border-border rounded-xl p-5">
                  {/* Goal Header */}
                  <div className="flex items-start gap-4 mb-5">
                    <div className={`p-2.5 ${model.iconBg} rounded-xl flex-shrink-0`}>
                      <Icon className={`w-5 h-5 ${model.iconColor}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap mb-1">
                        <h3 className="font-semibold text-foreground">{model.name}</h3>
                        <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30 text-xs">
                          In Development
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{model.goal}</p>
                    </div>
                    <div className="text-right flex-shrink-0 hidden sm:block">
                      <div className="text-lg font-bold text-foreground">{model.expectedAccuracy}</div>
                      <div className="text-xs text-muted-foreground">Expected Accuracy</div>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="mb-5">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs text-muted-foreground">{model.progressLabel}</span>
                      <span className="text-xs font-medium text-foreground">{model.progress}%</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-yellow-500 to-yellow-400/60"
                        style={{ width: `${model.progress}%` }}
                      />
                    </div>
                  </div>

                  {/* Approach steps */}
                  <div className="grid md:grid-cols-3 gap-4">
                    {model.approach.map((step, idx) => (
                      <div key={step.title} className="bg-secondary/30 border border-border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-5 h-5 rounded-full bg-primary/20 text-primary text-xs font-bold flex items-center justify-center flex-shrink-0">
                            {idx + 1}
                          </div>
                          <span className="text-sm font-semibold text-foreground">{step.title}</span>
                        </div>
                        <ul className="space-y-1.5">
                          {step.items.map((item) => (
                            <li key={item} className="flex items-start gap-2 text-xs text-muted-foreground">
                              <GitBranch className="w-3 h-3 text-primary mt-0.5 flex-shrink-0" />
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>

                  {/* Tech Stack */}
                  <div className="mt-4 pt-4 border-t border-border flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs text-muted-foreground">Planned stack:</span>
                      {["scikit-learn", "PyTorch", "PyDriller", "pandas"].map((t) => (
                        <span key={t} className="px-2 py-0.5 bg-secondary text-xs text-muted-foreground rounded-md border border-border">
                          {t}
                        </span>
                      ))}
                    </div>
                    <Button size="sm" variant="outline" className="gap-1.5 text-xs">
                      <FlaskConical className="w-3.5 h-3.5" />
                      View Research Notes
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
};

export default Models;
