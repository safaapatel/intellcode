import { useState, useMemo } from "react";
import { AppNavigation } from "@/components/app/AppNavigation";
import { Shield, Search, ChevronDown, ChevronUp, Settings2, Save, RotateCcw } from "lucide-react";
import { toast } from "sonner";

// ─── Rule data (sourced from backend/features/security_patterns.py) ───────────

interface Rule {
  id: string;
  title: string;
  description: string;
  severity: "critical" | "high" | "medium" | "low";
  cwe: string;
  confidence: number;
}

interface LangGroup {
  label: string;
  color: string;
  rules: Rule[];
}

const RULES: Record<string, LangGroup> = {
  python: {
    label: "Python",
    color: "text-blue-400",
    rules: [
      { id: "sql_injection", title: "SQL Injection via String Concatenation", description: "SQL query built by string concatenation allows attackers to inject arbitrary SQL. Use parameterized queries instead.", severity: "critical", cwe: "CWE-89", confidence: 0.92 },
      { id: "sql_injection", title: "SQL Injection via String Formatting", description: "SQL query built with str.format() or % formatting is vulnerable to injection. Use parameterized queries instead.", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "command_injection", title: "Command Injection via shell=True", description: "subprocess called with shell=True allows shell injection if any part of the command string is user-controlled.", severity: "critical", cwe: "CWE-78", confidence: 0.90 },
      { id: "insecure_deserialization", title: "Insecure Deserialization (pickle)", description: "pickle.loads/load on untrusted data can execute arbitrary code.", severity: "critical", cwe: "CWE-502", confidence: 0.88 },
      { id: "hardcoded_secret", title: "Hardcoded Credential", description: "Sensitive values embedded in source code can be exposed via version control. Use environment variables or a secrets manager.", severity: "critical", cwe: "CWE-798", confidence: 0.80 },
      { id: "hardcoded_secret", title: "Hardcoded AWS Access Key", description: "AWS access key (AKIA...) found in source. Revoke it immediately and use IAM roles or environment variables.", severity: "critical", cwe: "CWE-798", confidence: 0.95 },
      { id: "hardcoded_secret", title: "Embedded Private Key", description: "PEM private key block found in source. Store keys outside source control.", severity: "critical", cwe: "CWE-798", confidence: 0.95 },
      { id: "command_injection", title: "Command Injection via String Concat", description: "Command string built from string concatenation may be injectable.", severity: "high", cwe: "CWE-78", confidence: 0.70 },
      { id: "path_traversal", title: "Potential Path Traversal", description: "File path passed to open() may be user-controlled. Validate and sanitize paths before use.", severity: "high", cwe: "CWE-22", confidence: 0.65 },
      { id: "weak_cryptography", title: "Weak Hash Algorithm (MD5)", description: "Use of md5() is cryptographically broken. Use SHA-256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_cryptography", title: "Weak Hash Algorithm (SHA-1)", description: "Use of sha1() is cryptographically broken. Use SHA-256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_cryptography", title: "Weak Cipher (DES/RC4)", description: "DES and RC4 are cryptographically broken. Use AES-256-GCM.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "code_injection", title: "Use of eval()", description: "eval() on untrusted input allows arbitrary code execution.", severity: "high", cwe: "CWE-95", confidence: 0.75 },
      { id: "code_injection", title: "Use of exec()", description: "exec() on untrusted input allows arbitrary code execution.", severity: "high", cwe: "CWE-95", confidence: 0.75 },
      { id: "insecure_deserialization", title: "Insecure YAML Load", description: "yaml.load() without explicit Loader is unsafe. Use yaml.safe_load() instead.", severity: "high", cwe: "CWE-502", confidence: 0.85 },
      { id: "xxe", title: "Potential XML External Entity (XXE)", description: "Standard XML parsers are vulnerable to XXE attacks. Consider using defusedxml.", severity: "medium", cwe: "CWE-611", confidence: 0.60 },
      { id: "weak_cryptography", title: "Insecure Random", description: "random.random() is not cryptographically secure. Use secrets module.", severity: "medium", cwe: "CWE-338", confidence: 0.85 },
    ],
  },

  javascript: {
    label: "JavaScript / TypeScript",
    color: "text-yellow-400",
    rules: [
      { id: "xss", title: "DOM-based XSS via innerHTML", description: "Assigning to innerHTML with unsanitised input allows XSS. Use textContent or DOMPurify.", severity: "critical", cwe: "CWE-79", confidence: 0.85 },
      { id: "xss", title: "DOM-based XSS via outerHTML", description: "Assigning to outerHTML with unsanitised input allows XSS.", severity: "critical", cwe: "CWE-79", confidence: 0.85 },
      { id: "code_injection", title: "Use of eval()", description: "eval() on untrusted input allows arbitrary code execution.", severity: "critical", cwe: "CWE-95", confidence: 0.90 },
      { id: "sql_injection", title: "SQL Injection via String Concatenation", description: "SQL query built from string concatenation. Use parameterised queries.", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "xss", title: "XSS via document.write()", description: "document.write() with user-controlled input enables XSS attacks.", severity: "high", cwe: "CWE-79", confidence: 0.80 },
      { id: "code_injection", title: "Dynamic Function() Constructor", description: "new Function() with user input is equivalent to eval().", severity: "high", cwe: "CWE-95", confidence: 0.80 },
      { id: "code_injection", title: "setTimeout with String Argument", description: "Passing a string to setTimeout is equivalent to eval().", severity: "high", cwe: "CWE-95", confidence: 0.85 },
      { id: "code_injection", title: "setInterval with String Argument", description: "Passing a string to setInterval is equivalent to eval().", severity: "high", cwe: "CWE-95", confidence: 0.85 },
      { id: "weak_crypto", title: "Deprecated crypto.createCipher()", description: "crypto.createCipher() is deprecated. Use createCipheriv() with a random IV.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5/SHA-1)", description: "MD5/SHA-1 are cryptographically broken. Use SHA-256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.75 },
      { id: "prototype_pollution", title: "Prototype Pollution Risk", description: "Assigning to __proto__ can pollute the Object prototype.", severity: "high", cwe: "CWE-1321", confidence: 0.80 },
      { id: "path_traversal", title: "Potential Path Traversal", description: "File path built from concatenation may allow directory traversal.", severity: "high", cwe: "CWE-22", confidence: 0.70 },
      { id: "nosql_injection", title: "MongoDB $where Injection", description: "$where evaluates JavaScript; avoid with user-supplied values.", severity: "high", cwe: "CWE-943", confidence: 0.80 },
      { id: "open_redirect", title: "Potential Open Redirect", description: "Redirect target derived from request input. Validate the URL.", severity: "medium", cwe: "CWE-601", confidence: 0.70 },
      { id: "weak_crypto", title: "Insecure Random (Math.random)", description: "Math.random() is not cryptographically secure. Use crypto.getRandomValues().", severity: "medium", cwe: "CWE-338", confidence: 0.80 },
    ],
  },

  java: {
    label: "Java",
    color: "text-orange-400",
    rules: [
      { id: "sql_injection", title: "SQL Injection via String Concatenation", description: "Build SQL with PreparedStatement and bind parameters instead.", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "command_injection", title: "OS Command Execution via Runtime.exec()", description: "Runtime.exec() with user-controlled input allows command injection.", severity: "critical", cwe: "CWE-78", confidence: 0.90 },
      { id: "insecure_deserialization", title: "Unsafe Java Deserialization", description: "ObjectInputStream on untrusted data allows remote code execution.", severity: "critical", cwe: "CWE-502", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5/SHA-1)", description: "MD5 and SHA-1 are broken. Use SHA-256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.90 },
      { id: "weak_crypto", title: "Weak Cipher (DES/RC4/RC2)", description: "DES and RC4 are broken. Use AES-GCM.", severity: "high", cwe: "CWE-327", confidence: 0.90 },
      { id: "path_traversal", title: "Potential Path Traversal via Request Parameter", description: "File path derived from request input. Validate and canonicalize paths.", severity: "high", cwe: "CWE-22", confidence: 0.80 },
      { id: "hardcoded_credential", title: "Hardcoded Credential", description: "Credentials should not be stored in source code.", severity: "high", cwe: "CWE-798", confidence: 0.80 },
      { id: "command_injection", title: "ProcessBuilder Command Execution", description: "Ensure no user-controlled values are passed to ProcessBuilder.", severity: "high", cwe: "CWE-78", confidence: 0.70 },
      { id: "weak_crypto", title: "Insecure Random (java.util.Random)", description: "java.util.Random is not cryptographically secure. Use SecureRandom.", severity: "medium", cwe: "CWE-338", confidence: 0.80 },
      { id: "xxe", title: "XML Parser XXE Risk (DocumentBuilderFactory)", description: "Disable external entity processing: setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true).", severity: "medium", cwe: "CWE-611", confidence: 0.70 },
      { id: "xxe", title: "XML Parser XXE Risk (SAXParserFactory)", description: "Disable external DTD and entity processing on the SAX parser.", severity: "medium", cwe: "CWE-611", confidence: 0.70 },
      { id: "sql_injection", title: "Use of createStatement (prefer PreparedStatement)", description: "createStatement() with concatenated SQL is vulnerable to injection.", severity: "medium", cwe: "CWE-89", confidence: 0.65 },
    ],
  },

  go: {
    label: "Go",
    color: "text-cyan-400",
    rules: [
      { id: "insecure_tls", title: "TLS Certificate Verification Disabled", description: "Setting InsecureSkipVerify to true disables certificate validation.", severity: "critical", cwe: "CWE-295", confidence: 0.95 },
      { id: "sql_injection", title: "Potential SQL Injection via fmt.Sprintf", description: "Build queries with parameterised statements using database/sql placeholders.", severity: "high", cwe: "CWE-89", confidence: 0.75 },
      { id: "command_injection", title: "OS Command Execution via exec.Command", description: "Ensure exec.Command arguments are not derived from user-controlled input.", severity: "high", cwe: "CWE-78", confidence: 0.70 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5)", description: "MD5 is cryptographically broken. Use crypto/sha256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (SHA-1)", description: "SHA-1 is cryptographically broken. Use crypto/sha256 or stronger.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "path_traversal", title: "Potential Path Traversal (ioutil.ReadFile)", description: "File path built from concatenation may allow directory traversal.", severity: "high", cwe: "CWE-22", confidence: 0.70 },
      { id: "path_traversal", title: "Potential Path Traversal via os.Open", description: "Validate and sanitize file paths before opening.", severity: "medium", cwe: "CWE-22", confidence: 0.65 },
      { id: "insecure_config", title: "Default ServeMux with No Middleware", description: "Consider using a custom router with rate limiting and authentication.", severity: "medium", cwe: "CWE-284", confidence: 0.60 },
      { id: "weak_crypto", title: "Insecure Random (math/rand)", description: "math/rand is not cryptographically secure. Use crypto/rand.", severity: "medium", cwe: "CWE-338", confidence: 0.80 },
    ],
  },

  rust: {
    label: "Rust",
    color: "text-red-400",
    rules: [
      { id: "memory_safety", title: "Use of mem::transmute", description: "transmute reinterprets memory bits — can cause undefined behaviour.", severity: "critical", cwe: "CWE-704", confidence: 0.90 },
      { id: "command_injection", title: "OS Command Execution", description: "Ensure Command arguments are not derived from user-controlled input.", severity: "high", cwe: "CWE-78", confidence: 0.70 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5 crate)", description: "MD5 is cryptographically broken. Use sha2 or blake3.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "memory_safety", title: "Unsafe UTF-8 Conversion", description: "from_utf8_unchecked bypasses validity checks. Use from_utf8() instead.", severity: "high", cwe: "CWE-20", confidence: 0.85 },
      { id: "unsafe_code", title: "Unsafe Block", description: "Unsafe blocks bypass Rust's memory safety guarantees. Review carefully.", severity: "medium", cwe: "CWE-119", confidence: 0.80 },
      { id: "error_handling", title: "Unchecked unwrap() — Potential Panic", description: "Prefer explicit error handling with match or ? instead of unwrap().", severity: "low", cwe: "CWE-391", confidence: 0.60 },
      { id: "error_handling", title: "Unchecked expect() — Potential Panic", description: "expect() panics on None/Err. Use proper error propagation.", severity: "low", cwe: "CWE-391", confidence: 0.55 },
    ],
  },

  csharp: {
    label: "C#",
    color: "text-purple-400",
    rules: [
      { id: "sql_injection", title: "SQL Injection via String Concatenation", description: "Use SqlParameter or LINQ instead of concatenating SQL strings.", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "insecure_deserialization", title: "Unsafe Deserialisation via BinaryFormatter", description: "BinaryFormatter is insecure and deprecated. Use System.Text.Json or Newtonsoft.", severity: "critical", cwe: "CWE-502", confidence: 0.90 },
      { id: "command_injection", title: "OS Process Execution via Process.Start", description: "Validate all arguments passed to Process.Start.", severity: "high", cwe: "CWE-78", confidence: 0.75 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5)", description: "MD5 is broken. Use SHA-256 via SHA256.Create().", severity: "high", cwe: "CWE-327", confidence: 0.90 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (SHA-1)", description: "SHA-1 is broken. Use SHA-256 via SHA256.Create().", severity: "high", cwe: "CWE-327", confidence: 0.90 },
      { id: "xss", title: "Potential XSS via Response.Write", description: "Encode output using HttpUtility.HtmlEncode before writing to response.", severity: "high", cwe: "CWE-79", confidence: 0.80 },
      { id: "hardcoded_credential", title: "Hardcoded Credential", description: "Store credentials in environment variables or a secrets manager.", severity: "high", cwe: "CWE-798", confidence: 0.80 },
      { id: "weak_crypto", title: "Insecure Random (System.Random)", description: "System.Random is not cryptographically secure. Use RNGCryptoServiceProvider.", severity: "medium", cwe: "CWE-338", confidence: 0.80 },
      { id: "xxe", title: "XmlDocument XXE Risk", description: "Disable DTD processing: set XmlResolver = null.", severity: "medium", cwe: "CWE-611", confidence: 0.65 },
      { id: "sql_injection", title: "Direct SQL Execution — Verify Parameterisation", description: "Ensure all SQL commands use parameterised queries.", severity: "low", cwe: "CWE-89", confidence: 0.50 },
    ],
  },

  ruby: {
    label: "Ruby",
    color: "text-rose-400",
    rules: [
      { id: "sql_injection", title: "SQL Injection via String Interpolation", description: "Use parameterised queries or ActiveRecord finders instead of raw SQL.", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "command_injection", title: "OS Command Execution", description: "Avoid passing user-controlled data to system(), exec(), or backticks.", severity: "critical", cwe: "CWE-78", confidence: 0.85 },
      { id: "insecure_deserialization", title: "Unsafe Deserialisation via Marshal.load", description: "Marshal.load on untrusted data allows remote code execution.", severity: "critical", cwe: "CWE-502", confidence: 0.90 },
      { id: "code_injection", title: "Use of eval()", description: "eval() on untrusted input allows arbitrary code execution.", severity: "critical", cwe: "CWE-95", confidence: 0.88 },
      { id: "sql_injection", title: "SQL Injection in ActiveRecord where() Clause", description: "Use hash conditions or ? placeholders instead of interpolated strings.", severity: "high", cwe: "CWE-89", confidence: 0.85 },
      { id: "xss", title: "Potential XSS via render :inline", description: "render inline: evaluates ERB — sanitize user input before rendering.", severity: "high", cwe: "CWE-79", confidence: 0.75 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5)", description: "MD5 is broken. Use Digest::SHA256.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (SHA-1)", description: "SHA-1 is broken. Use Digest::SHA256.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "xss", title: "Possible XSS via html_safe / raw", description: "html_safe and raw mark strings as trusted — ensure they are sanitized.", severity: "medium", cwe: "CWE-79", confidence: 0.70 },
      { id: "weak_crypto", title: "Insecure Random (Kernel.rand)", description: "Kernel.rand is not cryptographically secure. Use SecureRandom.", severity: "medium", cwe: "CWE-338", confidence: 0.70 },
    ],
  },

  php: {
    label: "PHP",
    color: "text-indigo-400",
    rules: [
      { id: "sql_injection", title: "SQL Injection via String Concatenation (mysql_query)", description: "Use PDO or MySQLi with prepared statements.", severity: "critical", cwe: "CWE-89", confidence: 0.90 },
      { id: "sql_injection", title: "SQL Injection in mysqli_query", description: "Use prepared statements with bind_param().", severity: "critical", cwe: "CWE-89", confidence: 0.88 },
      { id: "xss", title: "XSS via Direct echo of Superglobal", description: "Always htmlspecialchars() or htmlentities() output from user input.", severity: "critical", cwe: "CWE-79", confidence: 0.90 },
      { id: "command_injection", title: "OS Command Execution", description: "Never pass user-controlled data to shell execution functions.", severity: "critical", cwe: "CWE-78", confidence: 0.88 },
      { id: "code_injection", title: "Code Injection via eval($...)", description: "eval() on dynamic data allows arbitrary PHP execution.", severity: "critical", cwe: "CWE-95", confidence: 0.90 },
      { id: "path_traversal", title: "Remote/Local File Inclusion via Variable", description: "Never include files based on user-controlled paths.", severity: "critical", cwe: "CWE-22", confidence: 0.88 },
      { id: "insecure_deserialization", title: "Unsafe Deserialisation via unserialize()", description: "unserialize() on untrusted data allows object injection. Use json_decode().", severity: "critical", cwe: "CWE-502", confidence: 0.90 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5)", description: "MD5 is cryptographically broken. Use password_hash() or hash('sha256', ...).", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (SHA-1)", description: "SHA-1 is broken. Use hash('sha256', ...) or password_hash().", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "open_redirect", title: "Open Redirect via User-Controlled Location Header", description: "Validate redirect targets against an allowlist.", severity: "high", cwe: "CWE-601", confidence: 0.80 },
      { id: "input_validation", title: "Unvalidated Superglobal Access", description: "Validate and sanitize all superglobal inputs before use.", severity: "medium", cwe: "CWE-20", confidence: 0.55 },
      { id: "weak_crypto", title: "Insecure Random (rand/mt_rand)", description: "rand() and mt_rand() are not cryptographically secure. Use random_bytes().", severity: "medium", cwe: "CWE-338", confidence: 0.75 },
    ],
  },

  c: {
    label: "C / C++",
    color: "text-gray-400",
    rules: [
      { id: "buffer_overflow", title: "Unsafe gets() — No Bounds Checking", description: "gets() is removed from C11. Use fgets() with explicit size limit.", severity: "critical", cwe: "CWE-120", confidence: 0.95 },
      { id: "use_after_free", title: "Double Free Detected", description: "Freeing the same pointer twice causes undefined behaviour. Set pointer to NULL after free.", severity: "critical", cwe: "CWE-415", confidence: 0.70 },
      { id: "buffer_overflow", title: "Unsafe sprintf — Potential Buffer Overflow", description: "Use snprintf() with explicit size to prevent buffer overflows.", severity: "high", cwe: "CWE-120", confidence: 0.85 },
      { id: "buffer_overflow", title: "Unsafe strcpy — No Bounds Checking", description: "Use strncpy() or strlcpy() with explicit size.", severity: "high", cwe: "CWE-120", confidence: 0.85 },
      { id: "buffer_overflow", title: "Unsafe strcat — No Bounds Checking", description: "Use strncat() with explicit size.", severity: "high", cwe: "CWE-120", confidence: 0.80 },
      { id: "command_injection", title: "OS Command Execution via system()", description: "Avoid system() with user input. Use execve() with validated args instead.", severity: "high", cwe: "CWE-78", confidence: 0.80 },
      { id: "format_string", title: "Format String Vulnerability", description: "Never pass untrusted input directly to printf. Use printf(\"%s\", str).", severity: "high", cwe: "CWE-134", confidence: 0.85 },
      { id: "weak_crypto", title: "Weak Hash Algorithm (MD5)", description: "MD5 is cryptographically broken. Use SHA-256 via OpenSSL EVP.", severity: "high", cwe: "CWE-327", confidence: 0.85 },
      { id: "sql_injection", title: "Potential SQL Injection in C", description: "Use parameterised ODBC queries instead of string concatenation.", severity: "high", cwe: "CWE-89", confidence: 0.65 },
      { id: "weak_crypto", title: "Insecure Random (rand())", description: "rand() is not cryptographically secure. Use /dev/urandom or platform CSPRNG.", severity: "medium", cwe: "CWE-338", confidence: 0.80 },
      { id: "memory_leak", title: "malloc Result Not Checked", description: "Always check the return value of malloc() for NULL.", severity: "low", cwe: "CWE-252", confidence: 0.55 },
      // C++ extras
      { id: "memory_safety", title: "Use of reinterpret_cast (C++)", description: "reinterpret_cast bypasses type safety. Review carefully.", severity: "medium", cwe: "CWE-704", confidence: 0.65 },
      { id: "memory_safety", title: "Use of const_cast (C++)", description: "const_cast removes const qualifier — may cause undefined behaviour if the original was const.", severity: "low", cwe: "CWE-704", confidence: 0.55 },
      { id: "memory_leak", title: "Potential Memory Leak — new without delete (C++)", description: "Prefer smart pointers (unique_ptr, shared_ptr) over raw new/delete.", severity: "low", cwe: "CWE-401", confidence: 0.45 },
    ],
  },
};

// ─── Severity helpers ──────────────────────────────────────────────────────────

const SEV_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };

const SEV_STYLES: Record<string, string> = {
  critical: "bg-red-500/15 text-red-400 border border-red-500/30",
  high:     "bg-orange-500/15 text-orange-400 border border-orange-500/30",
  medium:   "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  low:      "bg-blue-500/15 text-blue-400 border border-blue-500/30",
};

const SEV_DOT: Record<string, string> = {
  critical: "bg-red-500",
  high:     "bg-orange-500",
  medium:   "bg-yellow-500",
  low:      "bg-blue-500",
};

// ─── Rule card ────────────────────────────────────────────────────────────────

function RuleCard({ rule }: { rule: Rule }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="bg-secondary/30 border border-border rounded-lg overflow-hidden cursor-pointer hover:border-border/80 transition-colors"
      onClick={() => setOpen(o => !o)}
    >
      <div className="flex items-start gap-3 p-3">
        <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${SEV_DOT[rule.severity]}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-foreground">{rule.title}</span>
            <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full capitalize ${SEV_STYLES[rule.severity]}`}>
              {rule.severity}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-[11px] text-muted-foreground font-mono">{rule.cwe}</span>
            <span className="text-[11px] text-muted-foreground">
              {Math.round(rule.confidence * 100)}% confidence
            </span>
          </div>
        </div>
        <span className="text-muted-foreground flex-shrink-0 mt-0.5">
          {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </span>
      </div>
      {open && (
        <div className="px-4 pb-3 pt-0 text-xs text-muted-foreground border-t border-border/50 bg-secondary/20">
          <p className="mt-2 leading-relaxed">{rule.description}</p>
          <span className="inline-block mt-2 font-mono text-[10px] bg-secondary px-1.5 py-0.5 rounded text-foreground/60">
            {rule.id}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const SEVERITIES = ["all", "critical", "high", "medium", "low"] as const;

const SETTINGS_KEY = "intellcode_project_settings";

interface ProjectSettings {
  threshold: number;
  maxScoreDrop: number | null;
  failOnIssues: boolean;
  suppressedRules: string[];
  maxInlineComments: number;
}

const DEFAULT_SETTINGS: ProjectSettings = {
  threshold: 60,
  maxScoreDrop: null,
  failOnIssues: false,
  suppressedRules: [],
  maxInlineComments: 30,
};

function loadSettings(): ProjectSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(s: ProjectSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

function generateYml(s: ProjectSettings): string {
  const suppLines = s.suppressedRules.length
    ? s.suppressedRules.map(r => `  - ${r}`).join("\n")
    : "  []";
  return `# .intellicode.yml — generated from project settings
threshold: ${s.threshold}
fail_on_issues: ${s.failOnIssues}
max_score_drop: ${s.maxScoreDrop ?? "~"}
max_inline_comments: ${s.maxInlineComments}
suppress_rules:
${suppLines}
`;
}

function ProjectSettingsPanel() {
  const [settings, setSettings] = useState<ProjectSettings>(loadSettings);
  const [showYml, setShowYml] = useState(false);

  const save = () => {
    saveSettings(settings);
    toast.success("Project settings saved", { description: "Reflected in .intellicode.yml preview below." });
  };

  const reset = () => {
    setSettings(DEFAULT_SETTINGS);
    saveSettings(DEFAULT_SETTINGS);
    toast.info("Settings reset to defaults");
  };

  const allRuleIds = useMemo(() => {
    const ids = new Set<string>();
    Object.values(RULES).forEach(g => g.rules.forEach(r => ids.add(r.id)));
    return [...ids].sort();
  }, []);

  const toggleSuppress = (id: string) => {
    setSettings(s => ({
      ...s,
      suppressedRules: s.suppressedRules.includes(id)
        ? s.suppressedRules.filter(r => r !== id)
        : [...s.suppressedRules, id],
    }));
  };

  return (
    <div className="bg-secondary/20 border border-border rounded-xl p-5 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-primary" />
          <h2 className="font-semibold text-sm">Project Settings</h2>
        </div>
        <div className="flex gap-2">
          <button onClick={reset} className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground px-2.5 py-1.5 rounded-lg border border-border hover:border-border/80 transition-colors">
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
          <button onClick={save} className="flex items-center gap-1.5 text-xs font-medium text-primary bg-primary/10 hover:bg-primary/20 px-3 py-1.5 rounded-lg border border-primary/30 transition-colors">
            <Save className="w-3 h-3" /> Save
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Quality threshold */}
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Quality threshold (0–100)
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range" min={0} max={100} step={5}
              value={settings.threshold}
              onChange={e => setSettings(s => ({ ...s, threshold: Number(e.target.value) }))}
              className="flex-1 accent-primary"
            />
            <span className="w-8 text-right text-sm font-mono font-semibold">{settings.threshold}</span>
          </div>
          <p className="text-[11px] text-muted-foreground mt-1">PRs scoring below this get REQUEST_CHANGES</p>
        </div>

        {/* Max score drop */}
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Max score drop (delta gate)
          </label>
          <div className="flex items-center gap-2">
            <input
              type="number" min={0} max={50} placeholder="disabled"
              value={settings.maxScoreDrop ?? ""}
              onChange={e => setSettings(s => ({ ...s, maxScoreDrop: e.target.value === "" ? null : Number(e.target.value) }))}
              className="w-full text-sm bg-secondary/40 border border-border rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary/50"
            />
          </div>
          <p className="text-[11px] text-muted-foreground mt-1">Block PRs that drop quality by this many points</p>
        </div>

        {/* Max inline comments */}
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Max inline comments
          </label>
          <input
            type="number" min={1} max={30}
            value={settings.maxInlineComments}
            onChange={e => setSettings(s => ({ ...s, maxInlineComments: Number(e.target.value) }))}
            className="w-full text-sm bg-secondary/40 border border-border rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
        </div>

        {/* Fail on issues */}
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">CI behaviour</label>
          <button
            onClick={() => setSettings(s => ({ ...s, failOnIssues: !s.failOnIssues }))}
            className={`flex items-center gap-2 w-full text-sm px-3 py-1.5 rounded-lg border transition-colors ${
              settings.failOnIssues
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : "bg-secondary/40 border-border text-muted-foreground"
            }`}
          >
            <span className={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${settings.failOnIssues ? "bg-red-500 border-red-500" : "border-border"}`} />
            {settings.failOnIssues ? "Fail CI when below threshold" : "Report only (no CI failure)"}
          </button>
        </div>
      </div>

      {/* Suppressed rules */}
      <div>
        <label className="block text-xs font-medium text-muted-foreground mb-2">
          Globally suppressed rules ({settings.suppressedRules.length} suppressed)
        </label>
        <div className="flex flex-wrap gap-1.5">
          {allRuleIds.map(id => {
            const suppressed = settings.suppressedRules.includes(id);
            return (
              <button
                key={id}
                onClick={() => toggleSuppress(id)}
                className={`text-[11px] font-mono px-2 py-0.5 rounded border transition-colors ${
                  suppressed
                    ? "bg-red-500/10 border-red-500/30 text-red-400 line-through"
                    : "bg-secondary/40 border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                {id}
              </button>
            );
          })}
        </div>
      </div>

      {/* YAML preview */}
      <div>
        <button
          onClick={() => setShowYml(v => !v)}
          className="text-xs text-primary hover:underline flex items-center gap-1"
        >
          {showYml ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {showYml ? "Hide" : "Show"} .intellicode.yml preview
        </button>
        {showYml && (
          <div className="mt-2 relative">
            <pre className="text-[11px] font-mono bg-secondary/40 border border-border rounded-lg p-3 overflow-x-auto text-muted-foreground leading-relaxed">
              {generateYml(settings)}
            </pre>
            <button
              onClick={() => { navigator.clipboard.writeText(generateYml(settings)); toast.success("Copied to clipboard"); }}
              className="absolute top-2 right-2 text-[10px] text-muted-foreground hover:text-foreground px-2 py-0.5 rounded bg-secondary border border-border"
            >
              Copy
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function Rules() {
  const [activeLang, setActiveLang] = useState<string>("python");
  const [query, setQuery] = useState("");
  const [sevFilter, setSevFilter] = useState<string>("all");
  const [showSettings, setShowSettings] = useState(false);

  const group = RULES[activeLang];

  const filtered = useMemo(() => {
    let rules = group.rules;
    if (sevFilter !== "all") rules = rules.filter(r => r.severity === sevFilter);
    if (query.trim()) {
      const q = query.toLowerCase();
      rules = rules.filter(r =>
        r.title.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q) ||
        r.cwe.toLowerCase().includes(q)
      );
    }
    return [...rules].sort((a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity]);
  }, [group, sevFilter, query]);

  const counts = useMemo(() => {
    const c: Record<string, number> = { critical: 0, high: 0, medium: 0, low: 0 };
    group.rules.forEach(r => c[r.severity]++);
    return c;
  }, [group]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppNavigation />
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Security Rules</h1>
              <p className="text-sm text-muted-foreground">
                Rules IntelliCode actively checks — sourced from the analysis engine
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowSettings(v => !v)}
            className={`flex items-center gap-2 text-sm px-3 py-2 rounded-lg border transition-colors ${
              showSettings
                ? "bg-primary/10 border-primary/30 text-primary"
                : "border-border text-muted-foreground hover:text-foreground hover:border-border/80"
            }`}
          >
            <Settings2 className="w-4 h-4" />
            Project Settings
          </button>
        </div>

        {showSettings && (
          <div className="mb-6">
            <ProjectSettingsPanel />
          </div>
        )}

        <div className="flex gap-6">
          {/* Language sidebar */}
          <div className="w-48 flex-shrink-0">
            <div className="sticky top-6 space-y-0.5">
              {Object.entries(RULES).map(([key, grp]) => (
                <button
                  key={key}
                  onClick={() => { setActiveLang(key); setQuery(""); setSevFilter("all"); }}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex items-center justify-between ${
                    activeLang === key
                      ? "bg-primary/10 text-foreground font-medium"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/40"
                  }`}
                >
                  <span className={activeLang === key ? grp.color : ""}>{grp.label}</span>
                  <span className="text-xs text-muted-foreground">{grp.rules.length}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Main content */}
          <div className="flex-1 min-w-0">
            {/* Severity summary bar */}
            <div className="flex gap-3 mb-4 flex-wrap">
              {(["critical", "high", "medium", "low"] as const).map(sev => (
                <div key={sev} className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${SEV_DOT[sev]}`} />
                  <span className="text-xs text-muted-foreground capitalize">{sev}</span>
                  <span className="text-xs font-semibold text-foreground">{counts[sev]}</span>
                </div>
              ))}
              <span className="text-xs text-muted-foreground ml-auto">{group.rules.length} rules total</span>
            </div>

            {/* Search + filter bar */}
            <div className="flex gap-2 mb-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                <input
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Search rules, CWE IDs..."
                  className="w-full pl-9 pr-3 py-2 text-sm bg-secondary/40 border border-border rounded-lg focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/60"
                />
              </div>
              <div className="flex gap-1">
                {SEVERITIES.map(sev => (
                  <button
                    key={sev}
                    onClick={() => setSevFilter(sev)}
                    className={`px-2.5 py-1.5 text-xs rounded-lg capitalize transition-colors ${
                      sevFilter === sev
                        ? "bg-primary/10 text-foreground font-medium border border-primary/30"
                        : "bg-secondary/40 text-muted-foreground border border-transparent hover:border-border"
                    }`}
                  >
                    {sev}
                  </button>
                ))}
              </div>
            </div>

            {/* Rules list */}
            {filtered.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground text-sm">
                No rules match your filter.
              </div>
            ) : (
              <div className="space-y-2">
                {filtered.map((rule, i) => (
                  <RuleCard key={`${rule.id}-${i}`} rule={rule} />
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
