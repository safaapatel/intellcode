import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // GitHub Pages serves from /intellcode/ sub-path
  base: mode === "production" ? "/intellcode/" : "/",
  cacheDir: "./.vite-cache",
  server: {
    host: "::",
    port: 8080,
    watch: {
      ignored: ["**/venv/**", "**/node_modules/**", "**/checkpoints/**", "**/data/**"],
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // React core — changes rarely, long-lived cache
          "vendor-react": ["react", "react-dom", "react-router-dom"],
          // Charts — heaviest dependency (~400 kB)
          "vendor-charts": ["recharts"],
          // Radix UI + shadcn/ui
          "vendor-ui": [
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-select",
            "@radix-ui/react-tabs",
            "@radix-ui/react-tooltip",
            "@radix-ui/react-switch",
            "@radix-ui/react-label",
            "@radix-ui/react-slot",
          ],
          // Query + utility
          "vendor-misc": ["@tanstack/react-query", "sonner", "lucide-react", "clsx", "tailwind-merge"],
        },
      },
    },
    // Raise the warning threshold now that we have proper splitting
    chunkSizeWarningLimit: 600,
  },
}));
