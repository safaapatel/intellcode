import React, { useState, useEffect } from "react";
import { analyzeCodeStream } from "@/services/api";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Plus, Check, FileCode, Play } from "lucide-react";
import {
  mockRuleSets,
  mockCodeQualityRules,
  mockSecurityRules,
  mockCodeSmellRules,
} from "@/data/mockData";
import { toast } from "sonner";

const RULES_KEY = "intellcode_rules";

function loadRulesState() {
  try {
    const raw = localStorage.getItem(RULES_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveRulesState(state: object) {
  localStorage.setItem(RULES_KEY, JSON.stringify(state));
}

// Sample code designed to exercise all rule categories
const SAMPLE_CODE = `import os, subprocess

def compute_report(data, config, flag, extra, mode, debug, verbose):
    """Intentionally complex function for preview."""
    result = []
    for item in data:
        if item > 0:
            if config:
                if flag:
                    if extra:
                        if mode == "fast":
                            result.append(item * 2)
                        else:
                            result.append(item * 3)
                    else:
                        result.append(item)
                else:
                    result.append(0)
            else:
                result.append(-item)
        elif item == 0:
            result.append(None)
        else:
            result.append(item ** 2)
    if debug and verbose and len(result) > 100:
        print("debug:", result)
    return result

def get_user(db, username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return db.execute(query)

def run_cmd(user_input):
    os.system("ls " + user_input)

def process_items_a(items):
    output = []
    for i in items:
        if i > 10:
            output.append(i * 2)
    return output

def process_items_b(items):
    output = []
    for i in items:
        if i > 10:
            output.append(i * 2)
    return output

this_is_a_very_long_variable_name_that_makes_the_line_exceed_the_configured_maximum_line_length_threshold = 42
`;

const SEV_CLS: Record<string, string> = {
  critical: "bg-destructive/20 text-destructive border border-destructive/40",
  high: "bg-orange-500/20 text-orange-400 border border-orange-500/40",
  medium: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
  low: "bg-primary/20 text-primary border border-primary/40",
  info: "bg-secondary text-muted-foreground border border-border",
  none: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40",
};

const severityOptions = [
  { value: "low", label: "Low", color: "bg-primary" },
  { value: "medium", label: "Medium", color: "bg-warning" },
  { value: "high", label: "High", color: "bg-[hsl(25,95%,53%)]" },
  { value: "critical", label: "Critical", color: "bg-destructive" },
];

const Rules = () => {
  const saved = loadRulesState();
  const [ruleSets, setRuleSets] = useState(saved?.ruleSets ?? mockRuleSets);
  const [selectedRuleSet, setSelectedRuleSet] = useState(saved?.ruleSets?.[0] ?? mockRuleSets[0]);
  const [isDefault, setIsDefault] = useState(selectedRuleSet.isDefault);
  // Sync isDefault when selected rule set changes
  useEffect(() => { setIsDefault(selectedRuleSet.isDefault); }, [selectedRuleSet.id]);
  const [codeQualityRules, setCodeQualityRules] = useState(saved?.codeQualityRules ?? mockCodeQualityRules);
  const [securityRules, setSecurityRules] = useState(saved?.securityRules ?? mockSecurityRules);
  const [codeSmellRules, setCodeSmellRules] = useState(saved?.codeSmellRules ?? mockCodeSmellRules);
  const [previewFile, setPreviewFile] = useState("sample_analytics.py");
  const [previewCode, setPreviewCode] = useState(SAMPLE_CODE);
  const [previewResults, setPreviewResults] = useState<Array<{ text: string; sev: string }> | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newLanguage, setNewLanguage] = useState("Python");
  const [newTemplate, setNewTemplate] = useState("blank");

  const persist = (updates: {
    ruleSets?: typeof ruleSets;
    codeQualityRules?: typeof codeQualityRules;
    securityRules?: typeof securityRules;
    codeSmellRules?: typeof codeSmellRules;
  }) => {
    saveRulesState({
      ruleSets: updates.ruleSets ?? ruleSets,
      codeQualityRules: updates.codeQualityRules ?? codeQualityRules,
      securityRules: updates.securityRules ?? securityRules,
      codeSmellRules: updates.codeSmellRules ?? codeSmellRules,
    });
  };

  const handleCreate = () => {
    if (!newName.trim()) {
      toast.error("Please enter a name for the rule set");
      return;
    }
    const newSet = {
      id: String(Date.now()),
      name: newName.trim(),
      description: newDescription.trim() || "Custom rule set",
      language: newLanguage,
      isDefault: false,
      repositories: 0,
      issues: 0,
    };
    const updated = [...ruleSets, newSet];
    setRuleSets(updated);
    setSelectedRuleSet(newSet);
    persist({ ruleSets: updated });
    setCreateOpen(false);
    setNewName("");
    setNewDescription("");
    setNewLanguage("Python");
    setNewTemplate("blank");
    toast.success(`Rule set "${newSet.name}" created`);
  };

  const handleSaveChanges = () => {
    const updatedSets = ruleSets.map((rs) =>
      rs.id === selectedRuleSet.id ? { ...rs, isDefault } : rs
    );
    setRuleSets(updatedSets);
    persist({ ruleSets: updatedSets });
    toast.success("Rule set saved successfully");
  };

  const handleDuplicate = () => {
    const copy = { ...selectedRuleSet, id: String(Date.now()), name: `${selectedRuleSet.name} (copy)`, isDefault: false };
    const updated = [...ruleSets, copy];
    setRuleSets(updated);
    setSelectedRuleSet(copy);
    persist({ ruleSets: updated });
    toast.success("Rule set duplicated");
  };

  const handleDelete = () => {
    if (ruleSets.length <= 1) { toast.error("Cannot delete the only rule set"); return; }
    const updated = ruleSets.filter((rs) => rs.id !== selectedRuleSet.id);
    setRuleSets(updated);
    setSelectedRuleSet(updated[0]);
    persist({ ruleSets: updated });
    toast.info("Rule set deleted");
  };

  const handlePreview = async () => {
    setIsPreviewing(true);
    setPreviewResults(null);
    try {
      const result = await analyzeCodeStream(previewCode, previewFile || "preview.py", "python", () => {});
      const violations: Array<{ text: string; sev: string }> = [];

      // Code Quality Rules
      for (const rule of codeQualityRules.filter((r) => r.enabled)) {
        const rn = rule.name.toLowerCase();
        if (rn.includes("complexity") && result.complexity.cyclomatic > rule.threshold) {
          for (const fi of result.complexity.function_issues) {
            violations.push({ text: `'${fi.name}' — ${fi.issue}`, sev: rule.severity });
          }
          if (result.complexity.function_issues.length === 0) {
            violations.push({ text: `Cyclomatic complexity ${result.complexity.cyclomatic} exceeds threshold ${rule.threshold}`, sev: rule.severity });
          }
        }
        if ((rn.includes("line") || rn.includes("length")) && result.complexity.n_lines_over_80 > 0) {
          violations.push({ text: `${result.complexity.n_lines_over_80} line(s) exceed ${rule.threshold} characters`, sev: rule.severity });
        }
      }

      // Security Rules
      if (securityRules.some((r) => r.enabled)) {
        for (const finding of result.security.findings) {
          violations.push({ text: `Line ${finding.lineno}: [${finding.severity.toUpperCase()}] ${finding.title} — ${finding.description}`, sev: finding.severity });
        }
      }

      // Code Smell Rules
      if (codeSmellRules.some((r) => r.enabled)) {
        for (const s of result.refactoring.suggestions.slice(0, 5)) {
          violations.push({ text: `${s.title} (${s.effort_minutes} min effort)`, sev: s.priority ?? "medium" });
        }
        for (const d of result.dead_code.issues.slice(0, 3)) {
          violations.push({ text: `Dead code: ${d.title}`, sev: d.severity ?? "low" });
        }
        if (result.clones.clones.length > 0) {
          violations.push({ text: `${result.clones.clones.length} duplicate code block(s) detected`, sev: "medium" });
        }
      }

      const seen = new Set<string>();
      const unique = violations.filter((v) => { if (seen.has(v.text)) return false; seen.add(v.text); return true; });
      if (unique.length === 0) {
        unique.push({ text: "No violations found — code passes all active rules ✓", sev: "none" });
      }
      setPreviewResults(unique);
      toast.success(`Preview completed — ${unique.length} issue(s) found`);
    } catch {
      toast.error("Backend unreachable — it may be warming up. Try again in 30s.");
      setPreviewResults([{ text: "Error: Backend offline or warming up. Wait 30 seconds and try again.", sev: "critical" }]);
    } finally {
      setIsPreviewing(false);
    }
  };

  const toggleRule = (
    rules: typeof codeQualityRules,
    setRules: React.Dispatch<React.SetStateAction<typeof codeQualityRules>>,
    id: string,
    key: "codeQualityRules" | "securityRules" | "codeSmellRules"
  ) => {
    const updated = rules.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r));
    setRules(updated);
    persist({ [key]: updated });
  };

  const updateSeverity = (
    rules: typeof codeQualityRules,
    setRules: React.Dispatch<React.SetStateAction<typeof codeQualityRules>>,
    id: string,
    severity: string,
    key: "codeQualityRules" | "securityRules" | "codeSmellRules"
  ) => {
    const updated = rules.map((r) => (r.id === id ? { ...r, severity } : r));
    setRules(updated);
    persist({ [key]: updated });
  };

  const updateThreshold = (
    rules: typeof codeQualityRules,
    setRules: React.Dispatch<React.SetStateAction<typeof codeQualityRules>>,
    id: string,
    threshold: number,
    key: "codeQualityRules" | "securityRules" | "codeSmellRules"
  ) => {
    const updated = rules.map((r) => (r.id === id ? { ...r, threshold } : r));
    setRules(updated);
    persist({ [key]: updated });
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 min-h-[calc(100vh-60px)] bg-card border-r border-border p-4">
          <h2 className="text-sm font-semibold text-muted-foreground mb-4">Your Rule Sets</h2>
          <div className="space-y-2">
            {ruleSets.map((ruleSet) => (
              <button
                key={ruleSet.id}
                onClick={() => {
                  setSelectedRuleSet(ruleSet);
                  setIsDefault(ruleSet.isDefault);
                }}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  selectedRuleSet.id === ruleSet.id
                    ? "border-primary bg-primary/10"
                    : "border-transparent hover:bg-secondary"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground">{ruleSet.name}</span>
                  {ruleSet.isDefault && (
                    <span className="px-1.5 py-0.5 bg-primary/20 text-primary text-xs rounded">
                      default
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {ruleSet.repositories} repositories, {ruleSet.issues} issues
                </p>
              </button>
            ))}
          </div>
          <Button className="w-full mt-4 bg-gradient-primary" size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Create New Rule Set
          </Button>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8">
          {/* Header */}
          <div className="flex items-start justify-between mb-8">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-foreground">{selectedRuleSet.name}</h1>
                <span className="px-2 py-1 bg-primary/20 text-primary text-xs rounded">
                  {selectedRuleSet.language}
                </span>
              </div>
              <p className="text-muted-foreground mt-1">{selectedRuleSet.description}</p>
              <div className="flex items-center gap-2 mt-3">
                <Switch checked={isDefault} onCheckedChange={setIsDefault} />
                <span className="text-sm text-foreground">Set as default</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Applied to: {selectedRuleSet.repositories} repositories • {selectedRuleSet.issues} issues identified
              </p>
            </div>
          </div>

          {/* Code Quality Rules */}
          <div className="bg-card border border-border rounded-xl mb-6">
            <div className="px-6 py-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">Code Quality Rules</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border text-sm">
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Rule</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Enabled</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Severity</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {codeQualityRules.map((rule) => (
                    <tr key={rule.id} className="border-b border-border last:border-b-0 hover:bg-secondary/20">
                      <td className="px-6 py-4 text-foreground">{rule.name}</td>
                      <td className="px-6 py-4">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={() => toggleRule(codeQualityRules, setCodeQualityRules, rule.id, "codeQualityRules")}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(codeQualityRules, setCodeQualityRules, rule.id, v, "codeQualityRules")}
                        >
                          <SelectTrigger className="w-[120px] bg-input">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {severityOptions.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value}>
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            value={rule.threshold}
                            onChange={(e) =>
                              updateThreshold(codeQualityRules, setCodeQualityRules, rule.id, Number(e.target.value), "codeQualityRules")
                            }
                            className="w-20 bg-input"
                          />
                          <span className="text-sm text-muted-foreground">{rule.unit}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Security Rules */}
          <div className="bg-card border border-border rounded-xl mb-6">
            <div className="px-6 py-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">Security Rules</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border text-sm">
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Rule</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Enabled</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Severity</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Config</th>
                  </tr>
                </thead>
                <tbody>
                  {securityRules.map((rule) => (
                    <tr key={rule.id} className="border-b border-border last:border-b-0 hover:bg-secondary/20">
                      <td className="px-6 py-4 text-foreground">{rule.name}</td>
                      <td className="px-6 py-4">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={() => toggleRule(securityRules as any, setSecurityRules as any, rule.id, "securityRules")}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(securityRules as any, setSecurityRules as any, rule.id, v, "securityRules")}
                        >
                          <SelectTrigger className="w-[120px] bg-input">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {severityOptions.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value}>
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </td>
                      <td className="px-6 py-4 text-muted-foreground">—</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Code Smell Rules */}
          <div className="bg-card border border-border rounded-xl mb-6">
            <div className="px-6 py-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">Code Smell Rules</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border text-sm">
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Rule</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Enabled</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Severity</th>
                    <th className="text-left font-medium text-muted-foreground px-6 py-3">Config</th>
                  </tr>
                </thead>
                <tbody>
                  {codeSmellRules.map((rule) => (
                    <tr key={rule.id} className="border-b border-border last:border-b-0 hover:bg-secondary/20">
                      <td className="px-6 py-4 text-foreground">{rule.name}</td>
                      <td className="px-6 py-4">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={() => toggleRule(codeSmellRules as any, setCodeSmellRules as any, rule.id, "codeSmellRules")}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(codeSmellRules as any, setCodeSmellRules as any, rule.id, v, "codeSmellRules")}
                        >
                          <SelectTrigger className="w-[120px] bg-input">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {severityOptions.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value}>
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </td>
                      <td className="px-6 py-4">
                        {"threshold" in rule ? (
                          <div className="flex items-center gap-2">
                            <Input
                              type="number"
                              value={rule.threshold}
                              onChange={(e) =>
                                updateThreshold(codeSmellRules as any, setCodeSmellRules as any, rule.id, Number(e.target.value), "codeSmellRules")
                              }
                              className="w-20 bg-input"
                            />
                            <span className="text-sm text-muted-foreground">{rule.unit}</span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Preview Section */}
          <div className="bg-card border border-border rounded-xl p-6 mb-6">
            <h2 className="text-lg font-semibold text-foreground mb-1">Preview with Code</h2>
            <p className="text-xs text-muted-foreground mb-4">Edit the code below and run the active rules against it to see what would be flagged.</p>

            {/* Code editor chrome */}
            <div className="rounded-lg border border-border overflow-hidden mb-4">
              <div className="flex items-center justify-between bg-secondary/60 px-3 py-2 border-b border-border">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1.5">
                    <span className="w-3 h-3 rounded-full bg-red-500/70" />
                    <span className="w-3 h-3 rounded-full bg-yellow-500/70" />
                    <span className="w-3 h-3 rounded-full bg-emerald-500/70" />
                  </div>
                  <Input
                    value={previewFile}
                    onChange={(e) => setPreviewFile(e.target.value)}
                    placeholder="preview.py"
                    className="h-6 text-xs bg-transparent border-none p-0 font-mono text-muted-foreground w-48 focus-visible:ring-0"
                  />
                </div>
                <Button onClick={handlePreview} size="sm" disabled={isPreviewing} className="h-7 text-xs">
                  <Play className={`w-3.5 h-3.5 mr-1.5 ${isPreviewing ? "animate-spin" : ""}`} />
                  {isPreviewing ? "Analyzing…" : "Run Preview"}
                </Button>
              </div>
              <textarea
                value={previewCode}
                onChange={(e) => setPreviewCode(e.target.value)}
                spellCheck={false}
                className="w-full bg-[#0d1117] text-[#e6edf3] font-mono text-xs leading-5 p-4 resize-none outline-none min-h-[220px]"
                style={{ tabSize: 4 }}
              />
            </div>

            {previewResults && (
              <div className="bg-secondary/30 border border-border rounded-lg p-4">
                <p className="text-sm font-medium text-foreground mb-3">
                  {previewResults[0]?.sev === "none"
                    ? "All rules passed"
                    : `${previewResults.length} issue${previewResults.length !== 1 ? "s" : ""} found`}
                </p>
                <ul className="space-y-2">
                  {previewResults.map((v, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm font-mono">
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase shrink-0 mt-0.5 ${SEV_CLS[v.sev] ?? SEV_CLS.info}`}>
                        {v.sev === "none" ? "ok" : v.sev}
                      </span>
                      <span className="text-foreground">{v.text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button className="bg-gradient-primary" onClick={handleSaveChanges}>
              <Check className="w-4 h-4 mr-2" />
              Save Default Setting
            </Button>
            <Button variant="outline" onClick={handleDuplicate}>
              Duplicate Rule Set
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </div>
        </main>
      </div>

      {/* Create New Rule Set Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="bg-card border-border max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Rule Set</DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Name *</label>
              <Input
                placeholder="e.g. Python Strict, JS Standard..."
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="bg-input"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">Description</label>
              <Input
                placeholder="Brief description of this rule set"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                className="bg-input"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">Language</label>
                <Select value={newLanguage} onValueChange={setNewLanguage}>
                  <SelectTrigger className="bg-input">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["Python", "JavaScript", "TypeScript", "Go", "Java", "Multiple"].map((l) => (
                      <SelectItem key={l} value={l}>{l}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">Start From</label>
                <Select value={newTemplate} onValueChange={setNewTemplate}>
                  <SelectTrigger className="bg-input">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="blank">Blank</SelectItem>
                    <SelectItem value="python-strict">Python Strict</SelectItem>
                    <SelectItem value="js-standard">JS Standard</SelectItem>
                    <SelectItem value="security">Security Only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button className="bg-gradient-primary" onClick={handleCreate}>
              <Plus className="w-4 h-4 mr-2" />
              Create Rule Set
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Rules;
