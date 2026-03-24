/**
 * github.ts
 * Manages the GitHub OAuth access token and provides authenticated
 * GitHub API helpers used by Repositories.tsx and Settings.tsx.
 */

const TOKEN_KEY = "intellcode_github_token";
const USER_KEY = "intellcode_github_user";

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
  return localStorage.getItem(TOKEN_KEY);
}

export function setGitHubToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearGitHubToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isGitHubConnected(): boolean {
  return !!getGitHubToken();
}

// ─── Authenticated fetch helpers ──────────────────────────────────────────────

function authHeaders(): HeadersInit {
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

/** Fetch the authenticated user profile. Caches in localStorage. */
export async function getGitHubUser(forceRefresh = false): Promise<GitHubUser | null> {
  if (!getGitHubToken()) return null;
  if (!forceRefresh) {
    try {
      const cached = localStorage.getItem(USER_KEY);
      if (cached) return JSON.parse(cached) as GitHubUser;
    } catch { /* ignore */ }
  }
  const user = await ghFetch<GitHubUser>("https://api.github.com/user");
  localStorage.setItem(USER_KEY, JSON.stringify(user));
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

/** Fetch raw file content. */
export async function getRawFile(
  fullName: string,
  branch: string,
  path: string
): Promise<string> {
  const url = `https://raw.githubusercontent.com/${fullName}/${branch}/${path}`;
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Could not fetch ${path}`);
  return res.text();
}

/**
 * Read the GitHub OAuth token from the URL hash fragment after OAuth redirect.
 * Returns the token if found and cleans the URL, otherwise returns null.
 */
export function consumeTokenFromHash(): string | null {
  const hash = window.location.hash;
  const match = hash.match(/github_token=([^&]+)/);
  if (!match) return null;
  const token = match[1];
  // Clean the token from the URL without triggering a navigation
  const cleanHash = hash.replace(/github_token=[^&]+&?/, "").replace(/#$/, "");
  window.history.replaceState(null, "", window.location.pathname + window.location.search + (cleanHash || ""));
  return token;
}
