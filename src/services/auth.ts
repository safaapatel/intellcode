export const USERS_KEY = "intellcode_users";
export const SESSION_KEY = "intellcode_session";

export interface StoredUser {
  id: string;
  name: string;
  email: string;
  password: string;
  role: "admin" | "reviewer" | "developer";
  active: boolean;
  createdAt: string;
}

export interface Session {
  userId: string;
  name: string;
  email: string;
  role: "admin" | "reviewer" | "developer";
  loggedInAt: string;
}

// Demo password is intentionally generic — no real credentials stored here
export const DEMO_PASSWORD = "demo";

export const DEFAULT_USERS: StoredUser[] = [
  {
    id: "1",
    name: "Safaa Patel",
    email: "admin@intellicode.io",
    password: DEMO_PASSWORD,
    role: "admin",
    active: true,
    createdAt: "2025-11-01T00:00:00Z",
  },
  {
    id: "2",
    name: "Alex Reviewer",
    email: "reviewer@intellicode.io",
    password: DEMO_PASSWORD,
    role: "reviewer",
    active: true,
    createdAt: "2025-11-15T00:00:00Z",
  },
  {
    id: "3",
    name: "Dev User",
    email: "dev@intellicode.io",
    password: DEMO_PASSWORD,
    role: "developer",
    active: true,
    createdAt: "2025-12-01T00:00:00Z",
  },
];

export function getUsers(): StoredUser[] {
  try {
    const raw = localStorage.getItem(USERS_KEY);
    if (!raw) {
      localStorage.setItem(USERS_KEY, JSON.stringify(DEFAULT_USERS));
      return DEFAULT_USERS;
    }
    return JSON.parse(raw);
  } catch {
    return DEFAULT_USERS;
  }
}

export function saveUsers(users: StoredUser[]) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function getSession(): Session | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveSession(user: StoredUser) {
  const session: Session = {
    userId: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
    loggedInAt: new Date().toISOString(),
  };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  // Sync with intellcode_settings so AppNavigation picks up the name/email/role
  const existing = JSON.parse(localStorage.getItem("intellcode_settings") ?? "{}");
  localStorage.setItem(
    "intellcode_settings",
    JSON.stringify({ ...existing, name: user.name, email: user.email, role: user.role })
  );
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  // Remove user-specific fields from settings so the next user starts clean
  try {
    const raw = localStorage.getItem("intellcode_settings");
    if (raw) {
      const settings = JSON.parse(raw);
      delete settings.name;
      delete settings.email;
      delete settings.role;
      localStorage.setItem("intellcode_settings", JSON.stringify(settings));
    }
  } catch {
    // ignore
  }
}

export function generatePassword(): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$";
  const arr = new Uint8Array(12);
  crypto.getRandomValues(arr);
  return Array.from(arr)
    .map((b) => chars[b % chars.length])
    .join("");
}
