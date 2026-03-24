export type Theme = "dark" | "light" | "system";

const KEY = "intellcode_theme";

export function getTheme(): Theme {
  return (localStorage.getItem(KEY) as Theme) ?? "dark";
}

export function applyTheme(theme: Theme) {
  const resolved =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;
  document.documentElement.classList.toggle("light", resolved === "light");
  localStorage.setItem(KEY, theme);
  window.dispatchEvent(new CustomEvent("intellcode-theme", { detail: theme }));
}

export function initTheme() {
  applyTheme(getTheme());
}
