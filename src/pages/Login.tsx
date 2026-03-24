import { useState } from "react";
import { Link, useNavigate, useLocation, Navigate } from "react-router-dom";
import { Code2, Github, Loader2, Shield, Zap, GitBranch, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { getUsers, saveSession, getSession } from "@/services/auth";

type LegalModal = null | "terms" | "privacy";

const TERMS_CONTENT = `By using IntelliCode Review you agree to: (1) use the platform for lawful code analysis only; (2) not attempt to reverse-engineer the ML models; (3) keep your API token confidential; (4) accept that analysis results are advisory and not a warranty of code correctness. We may update these terms at any time.`;

const PRIVACY_CONTENT = `IntelliCode Review collects: code snippets you submit for analysis (processed server-side, not stored permanently), your GitHub username (for display), and anonymous usage metrics. We do not sell your data. Code snippets are deleted from memory after analysis completes. You may request deletion of your account data at any time via Settings → Danger Zone.`;

const Login = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname;
  const [loading, setLoading] = useState(false);
  const [legalModal, setLegalModal] = useState<LegalModal>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  if (getSession()) return <Navigate to="/dashboard" replace />;

  const handleGitHubLogin = () => {
    setLoading(true);
    toast.info("Connecting to GitHub...");
    setTimeout(() => {
      // Simulate GitHub login as admin user
      const users = getUsers();
      const admin = users.find((u) => u.role === "admin" && u.active);
      if (admin) {
        saveSession(admin);
        toast.success(`Welcome back, ${admin.name.split(" ")[0]}!`, {
          description: "Signed in via GitHub",
        });
        navigate("/dashboard");
      } else {
        setLoading(false);
        toast.error("No admin account found.");
      }
    }, 1800);
  };

  const handleEmailLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    if (!email.trim() || !password.trim()) {
      setLoginError("Please enter both email and password.");
      return;
    }
    const users = getUsers();
    const user = users.find(
      (u) => u.email.toLowerCase() === email.toLowerCase() && u.password === password
    );
    if (!user) {
      setLoginError("Invalid email or password.");
      return;
    }
    if (!user.active) {
      setLoginError("Your account has been deactivated. Contact an admin.");
      return;
    }
    saveSession(user);
    toast.success(`Welcome back, ${user.name.split(" ")[0]}!`, {
      description: `Signed in as ${user.role}`,
    });
    navigate(user.role === "admin" ? "/admin" : "/dashboard");
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Legal Modal */}
      {legalModal && (
        <div
          className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
          onClick={() => setLegalModal(null)}
        >
          <div
            className="bg-card border border-border rounded-xl p-6 max-w-md w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="font-semibold text-foreground mb-3">
              {legalModal === "terms" ? "Terms of Service" : "Privacy Policy"}
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed mb-5">
              {legalModal === "terms" ? TERMS_CONTENT : PRIVACY_CONTENT}
            </p>
            <button
              className="text-sm text-primary hover:underline"
              onClick={() => setLegalModal(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-card via-secondary/30 to-card border-r border-border flex-col items-center justify-center p-12 relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `linear-gradient(hsl(var(--primary)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--primary)) 1px, transparent 1px)`,
            backgroundSize: "40px 40px",
          }}
        />
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-primary/10 rounded-full blur-3xl" />

        <div className="relative z-10 text-center max-w-md">
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="p-3 bg-gradient-primary rounded-xl shadow-glow-primary">
              <Code2 className="w-8 h-8 text-background" />
            </div>
            <span className="text-2xl font-bold text-foreground">IntelliCode Review</span>
          </div>

          <h2 className="text-3xl font-bold text-foreground mb-4">AI-Powered Code Review</h2>
          <p className="text-muted-foreground text-lg mb-10">
            Catch bugs before they ship. IntelliCode Review uses machine learning to detect
            vulnerabilities, code smells, and quality issues automatically.
          </p>

          <div className="space-y-4 text-left">
            {[
              { icon: Shield, label: "Security Scanning", desc: "SQL injection, hardcoded secrets, path traversal" },
              { icon: Zap, label: "Instant Analysis", desc: "Under 10 seconds for most repositories" },
              { icon: GitBranch, label: "GitHub Integration", desc: "Reviews triggered on every pull request" },
            ].map(({ icon: Icon, label, desc }) => (
              <div key={label} className="flex items-start gap-3 p-3 bg-secondary/30 rounded-lg border border-border">
                <div className="p-1.5 bg-primary/20 rounded-md mt-0.5">
                  <Icon className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <div className="font-medium text-foreground text-sm">{label}</div>
                  <div className="text-xs text-muted-foreground">{desc}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-10 flex items-center justify-center gap-8">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">95%</div>
              <div className="text-xs text-muted-foreground">Bug Detection</div>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">60%</div>
              <div className="text-xs text-muted-foreground">Time Saved</div>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">10x</div>
              <div className="text-xs text-muted-foreground">Faster Reviews</div>
            </div>
          </div>

          {/* Demo credentials hint */}
          <div className="mt-8 p-4 bg-secondary/40 rounded-xl border border-border text-left">
            <p className="text-xs font-semibold text-muted-foreground mb-2">Demo access</p>
            <p className="text-xs text-muted-foreground font-mono">Any email · password: <span className="text-foreground">demo</span></p>
          </div>
        </div>
      </div>

      {/* Right Panel - Auth Form */}
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="flex lg:hidden items-center gap-2 mb-10">
          <div className="p-2 bg-gradient-primary rounded-lg">
            <Code2 className="w-5 h-5 text-background" />
          </div>
          <span className="text-xl font-bold text-foreground">IntelliCode Review</span>
        </div>

        <div className="w-full max-w-sm">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-foreground mb-2">Sign in to your account</h1>
            <p className="text-muted-foreground">
              Use your credentials or connect with GitHub
            </p>
          </div>

          {/* GitHub Login Button */}
          <Button
            className="w-full h-12 bg-[#24292e] hover:bg-[#1a1e22] text-white border border-[#30363d] text-base font-medium gap-3 mb-6"
            onClick={handleGitHubLogin}
            disabled={loading}
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Github className="w-5 h-5" />}
            {loading ? "Authenticating..." : "Continue with GitHub"}
          </Button>

          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-xs text-muted-foreground bg-background px-2">
              or sign in with email
            </div>
          </div>

          {/* Email / Password Form */}
          <form onSubmit={handleEmailLogin} className="space-y-4 mb-6">
            <div>
              <Label className="text-foreground mb-1.5 block">Email</Label>
              <Input
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setLoginError(null); }}
                placeholder="you@intellicode.io"
                className="bg-input border-border h-11"
                autoComplete="email"
              />
            </div>
            <div>
              <Label className="text-foreground mb-1.5 block">Password</Label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setLoginError(null); }}
                  placeholder="••••••••"
                  className="bg-input border-border h-11 pr-10"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  onClick={() => setShowPassword((v) => !v)}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {loginError && (
              <p className="text-sm text-destructive">{loginError}</p>
            )}

            <Button
              type="submit"
              className="w-full h-11 bg-gradient-primary text-base font-medium"
              disabled={loading}
            >
              Sign In
            </Button>
          </form>

          <p className="text-xs text-muted-foreground text-center leading-relaxed">
            By signing in, you agree to our{" "}
            <span className="text-primary cursor-pointer hover:underline" onClick={() => setLegalModal("terms")}>
              Terms of Service
            </span>{" "}
            and{" "}
            <span className="text-primary cursor-pointer hover:underline" onClick={() => setLegalModal("privacy")}>
              Privacy Policy
            </span>
            .
          </p>

          <div className="mt-6 text-center">
            <span className="text-sm text-muted-foreground">Don't have an account? </span>
            <Link to="/" className="text-sm text-primary hover:underline font-medium">
              Learn more
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
