import { useState } from "react";
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
import { Plus, Check, FileCode, Play } from "lucide-react";
import {
  mockRuleSets,
  mockCodeQualityRules,
  mockSecurityRules,
  mockCodeSmellRules,
} from "@/data/mockData";
import { toast } from "sonner";

const severityOptions = [
  { value: "low", label: "Low", color: "bg-primary" },
  { value: "medium", label: "Medium", color: "bg-warning" },
  { value: "high", label: "High", color: "bg-[hsl(25,95%,53%)]" },
  { value: "critical", label: "Critical", color: "bg-destructive" },
];

const Rules = () => {
  const [selectedRuleSet, setSelectedRuleSet] = useState(mockRuleSets[0]);
  const [isDefault, setIsDefault] = useState(selectedRuleSet.isDefault);
  const [codeQualityRules, setCodeQualityRules] = useState(mockCodeQualityRules);
  const [securityRules, setSecurityRules] = useState(mockSecurityRules);
  const [codeSmellRules, setCodeSmellRules] = useState(mockCodeSmellRules);
  const [previewFile, setPreviewFile] = useState("sample_analytics.py");
  const [previewResults, setPreviewResults] = useState<string[] | null>(null);

  const handleSaveChanges = () => {
    toast.success("Rule set saved successfully");
  };

  const handleDuplicate = () => {
    toast.success("Rule set duplicated");
  };

  const handleDelete = () => {
    toast.error("Rule set deleted");
  };

  const handlePreview = () => {
    setPreviewResults([
      "Line 45: Complexity exceeds threshold (15)",
      "Line 87: Line too long (96 characters)",
    ]);
    toast.info("Preview completed - 2 issues found");
  };

  const toggleRule = (
    rules: typeof codeQualityRules,
    setRules: typeof setCodeQualityRules,
    id: string
  ) => {
    setRules(rules.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r)));
  };

  const updateSeverity = (
    rules: typeof codeQualityRules,
    setRules: typeof setCodeQualityRules,
    id: string,
    severity: string
  ) => {
    setRules(rules.map((r) => (r.id === id ? { ...r, severity } : r)));
  };

  const updateThreshold = (
    rules: typeof codeQualityRules,
    setRules: typeof setCodeQualityRules,
    id: string,
    threshold: number
  ) => {
    setRules(rules.map((r) => (r.id === id ? { ...r, threshold } : r)));
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavigation />

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 min-h-[calc(100vh-60px)] bg-card border-r border-border p-4">
          <h2 className="text-sm font-semibold text-muted-foreground mb-4">Your Rule Sets</h2>
          <div className="space-y-2">
            {mockRuleSets.map((ruleSet) => (
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
          <Button className="w-full mt-4 bg-gradient-primary" size="sm">
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
                          onCheckedChange={() => toggleRule(codeQualityRules, setCodeQualityRules, rule.id)}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(codeQualityRules, setCodeQualityRules, rule.id, v)}
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
                              updateThreshold(codeQualityRules, setCodeQualityRules, rule.id, Number(e.target.value))
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
                          onCheckedChange={() => toggleRule(securityRules as any, setSecurityRules as any, rule.id)}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(securityRules as any, setSecurityRules as any, rule.id, v)}
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
                          onCheckedChange={() => toggleRule(codeSmellRules as any, setCodeSmellRules as any, rule.id)}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <Select
                          value={rule.severity}
                          onValueChange={(v) => updateSeverity(codeSmellRules as any, setCodeSmellRules as any, rule.id, v)}
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
                                updateThreshold(codeSmellRules as any, setCodeSmellRules as any, rule.id, Number(e.target.value))
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
            <h2 className="text-lg font-semibold text-foreground mb-4">Preview with Sample Code</h2>
            <div className="flex items-center gap-4 mb-4">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Choose File:</span>
              </div>
              <Input
                value={previewFile}
                onChange={(e) => setPreviewFile(e.target.value)}
                className="flex-1 max-w-md bg-input font-mono"
              />
              <Button onClick={handlePreview} size="sm">
                <Play className="w-4 h-4 mr-2" />
                Preview
              </Button>
            </div>
            {previewResults && (
              <div className="bg-secondary/30 border border-border rounded-lg p-4">
                <p className="text-sm font-medium text-foreground mb-2">
                  Results: {previewResults.length} issues found
                </p>
                <ul className="space-y-1">
                  {previewResults.map((result, index) => (
                    <li key={index} className="text-sm text-primary font-mono">
                      • {result}
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
              Save Changes
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
    </div>
  );
};

export default Rules;
