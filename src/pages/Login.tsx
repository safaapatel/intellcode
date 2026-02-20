import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Code2, Github, Loader2, Shield, Zap, GitBranch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Login = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleGitHubLogin = () => {
    setLoading(true);
    toast.info("Connecting to GitHub...");
    setTimeout(() => {
      toast.success("Authentication successful! Welcome back, Safaa.");
      navigate("/dashboard");
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-card via-secondary/30 to-card border-r border-border flex-col items-center justify-center p-12 relative overflow-hidden">
        {/* Background grid */}
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `linear-gradient(hsl(var(--primary)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--primary)) 1px, transparent 1px)`,
            backgroundSize: "40px 40px",
          }}
        />
        {/* Glow effect */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-primary/10 rounded-full blur-3xl" />

        <div className="relative z-10 text-center max-w-md">
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="p-3 bg-gradient-primary rounded-xl shadow-glow-primary">
              <Code2 className="w-8 h-8 text-background" />
            </div>
            <span className="text-2xl font-bold text-foreground">IntelliCode Review</span>
          </div>

          <h2 className="text-3xl font-bold text-foreground mb-4">
            AI-Powered Code Review
          </h2>
          <p className="text-muted-foreground text-lg mb-10">
            Catch bugs before they ship. IntelliCode Review uses machine learning to detect vulnerabilities, code smells, and quality issues automatically.
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
        </div>
      </div>

      {/* Right Panel - Auth Form */}
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        {/* Mobile logo */}
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
              Connect with GitHub to start reviewing code with AI
            </p>
          </div>

          {/* GitHub Login Button */}
          <Button
            className="w-full h-12 bg-[#24292e] hover:bg-[#1a1e22] text-white border border-[#30363d] text-base font-medium gap-3 mb-4"
            onClick={handleGitHubLogin}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Github className="w-5 h-5" />
            )}
            {loading ? "Authenticating..." : "Continue with GitHub"}
          </Button>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-xs text-muted-foreground bg-background px-2">
              or
            </div>
          </div>

          {/* Google Login Button (secondary) */}
          <Button
            variant="outline"
            className="w-full h-12 gap-3 mb-6 text-base"
            onClick={() => {
              toast.info("Google OAuth coming soon");
            }}
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Continue with Google
          </Button>

          <p className="text-xs text-muted-foreground text-center leading-relaxed">
            By signing in, you agree to our{" "}
            <span className="text-primary cursor-pointer hover:underline">Terms of Service</span>{" "}
            and{" "}
            <span className="text-primary cursor-pointer hover:underline">Privacy Policy</span>.
          </p>

          <div className="mt-8 text-center">
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
