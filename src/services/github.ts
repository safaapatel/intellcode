/**
 * github.ts
 * Manages the GitHub OAuth access token and provides authenticated
 * GitHub API helpers used by Repositories.tsx and Settings.tsx.
 */

const TOKEN_KEY = "intellcode_github_token";
const USER_KEY = "intellcode_github_user";

// GitHub OAuth tokens are stored in localStorage so the connection persists across
// sessions without requiring the user to reconnect every time.
const _store = localStorage;

// One-time migration: move any existing token from sessionStorage to localStorage.
(function migrateLegacyToken() {
  try {
    const tok = sessionStorage.getItem(TOKEN_KEY);
    const usr = sessionStorage.getItem(USER_KEY);
    if (tok && !localStorage.getItem(TOKEN_KEY)) {
      localStorage.setItem(TOKEN_KEY, tok);
      if (usr) localStorage.setItem(USER_KEY, usr);
    }
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
  } catch { /* ignore */ }
})();

export interface GitHubUser {
  login: string;
  name: string | null;
  avatar_url: string;
  public_repos: number;
  private_repos: number;
  html_url: string;
}

export interface GitHubRepo {
  id: number;
  name: string;
  full_name: string;
  description: string | null;
  private: boolean;
  language: string | null;
  stargazers_count: number;
  default_branch: string;
  html_url: string;
  updated_at: string;
}

// ─── Token storage ────────────────────────────────────────────────────────────

export function getGitHubToken(): string | null {
  return _store.getItem(TOKEN_KEY);
}

export function setGitHubToken(token: string) {
  _store.setItem(TOKEN_KEY, token);
}

export function clearGitHubToken() {
  _store.removeItem(TOKEN_KEY);
  _store.removeItem(USER_KEY);
}

export function isGitHubConnected(): boolean {
  return !!getGitHubToken();
}

// ─── Authenticated fetch helpers ──────────────────────────────────────────────

export function authHeaders(): HeadersInit {
  const token = getGitHubToken();
  return token
    ? { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" }
    : { Accept: "application/vnd.github+json" };
}

async function ghFetch<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) {
    if (res.status === 401) {
      clearGitHubToken();
      throw new Error("GitHub token expired — please reconnect");
    }
    throw new Error(`GitHub API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ─── API calls ────────────────────────────────────────────────────────────────

/** Fetch the authenticated user profile. Caches in sessionStorage. */
export async function getGitHubUser(forceRefresh = false): Promise<GitHubUser | null> {
  if (!getGitHubToken()) return null;
  if (!forceRefresh) {
    try {
      const cached = _store.getItem(USER_KEY);
      if (cached) return JSON.parse(cached) as GitHubUser;
    } catch { /* ignore */ }
  }
  const user = await ghFetch<GitHubUser>("https://api.github.com/user");
  _store.setItem(USER_KEY, JSON.stringify(user));
  return user;
}

/** List all repos the authenticated user can access (handles pagination). */
export async function listUserRepos(page = 1, perPage = 30): Promise<GitHubRepo[]> {
  return ghFetch<GitHubRepo[]>(
    `https://api.github.com/user/repos?per_page=${perPage}&page=${page}&sort=updated&affiliation=owner,collaborator`
  );
}

/** Fetch public repo info (works without token). */
export async function getRepoInfo(fullName: string): Promise<GitHubRepo> {
  return ghFetch<GitHubRepo>(`https://api.github.com/repos/${fullName}`);
}

/** Fetch the recursive file tree for a repo+branch. */
export async function getRepoTree(
  fullName: string,
  branch: string
): Promise<{ tree: Array<{ type: string; path: string; size?: number }> }> {
  return ghFetch(
    `https://api.github.com/repos/${fullName}/git/trees/${branch}?recursive=1`
  );
}

/** Fetch raw file content. Falls back to Contents API if CDN returns non-200. */
export async function getRawFile(
  fullName: string,
  branch: string,
  path: string
): Promise<string> {
  const token = getGitHubToken();
  const rawHeaders: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {};

  // Primary: raw CDN (fast, no rate-limit penalty)
  const rawRes = await fetch(
    `https://raw.githubusercontent.com/${fullName}/${branch}/${encodeURI(path)}`,
    { headers: rawHeaders }
  );
  if (rawRes.ok) return rawRes.text();

  // Fallback: Contents API (handles private repos & auth edge-cases)
  const apiRes = await fetch(
    `https://api.github.com/repos/${fullName}/contents/${encodeURI(path)}?ref=${branch}`,
    { headers: authHeaders() }
  );
  if (!apiRes.ok) throw new Error(`Could not fetch ${path}: ${rawRes.status}`);
  const data = await apiRes.json() as { content?: string; encoding?: string };
  if (data.content && data.encoding === "base64")
    return atob(data.content.replace(/\n/g, ""));
  throw new Error(`Could not fetch ${path}: unexpected format`);
}

/**
 * Read the GitHub OAuth token from the URL fragment after OAuth redirect.
 * The backend sends the token as #github_token=... (fragment) so it is never
 * transmitted to servers or recorded in proxy logs.
 * Returns the token if found and immediately cleans the fragment from the URL.
 */
export function consumeTokenFromHash(): string | null {
  const fragment = window.location.hash.slice(1); // strip leading #
  const params = new URLSearchParams(fragment);
  const token = params.get("github_token");
  if (!token) return null;
  // Remove github_token from fragment without triggering a navigation.
  params.delete("github_token");
  const newHash = params.toString() ? `#${params.toString()}` : "";
  window.history.replaceState(null, "", window.location.pathname + window.location.search + newHash);
  return token;
}
