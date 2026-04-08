export const USERS_KEY = "intellcode_users";
export const SESSION_KEY = "intellcode_session";

export interface StoredUser {
  id: string;
  name: string;
  email: string;
  /** PBKDF2-SHA256 hex digest. Value "needs_migration" means the record predates hashing. */
  passwordHash: string;
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

/** The plaintext password used for all demo accounts — never stored as-is. */
export const DEMO_PASSWORD = "demo";

/** Sentinel stored in passwordHash while a record awaits its first PBKDF2 migration. */
const MIGRATION_SENTINEL = "needs_migration";

/**
 * Hash a password with PBKDF2-SHA256.
 * The salt combines a fixed application prefix with the user's id so that two
 * users with the same password produce different hashes.
 */
export async function hashPassword(password: string, userId: string): Promise<string> {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    "raw", enc.encode(password), "PBKDF2", false, ["deriveBits"]
  );
  const bits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      hash: "SHA-256",
      salt: enc.encode(`intellcode:${userId}`),
      iterations: 100_000,
    },
    keyMaterial,
    256
  );
  return Array.from(new Uint8Array(bits))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// DEFAULT_USERS uses the migration sentinel — passwords are hashed on first login.
export const DEFAULT_USERS: StoredUser[] = [
  {
    id: "1",
    name: "Safaa Patel",
    email: "admin@intellicode.io",
    passwordHash: MIGRATION_SENTINEL,
    role: "admin",
    active: true,
    createdAt: "2025-11-01T00:00:00Z",
  },
  {
    id: "2",
    name: "Alex Reviewer",
    email: "reviewer@intellicode.io",
    passwordHash: MIGRATION_SENTINEL,
    role: "reviewer",
    active: true,
    createdAt: "2025-11-15T00:00:00Z",
  },
  {
    id: "3",
    name: "Dev User",
    email: "dev@intellicode.io",
    passwordHash: MIGRATION_SENTINEL,
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
    // Transparently handle old records that stored plaintext in a "password" field.
    const parsed = JSON.parse(raw) as (StoredUser & { password?: string })[];
    let needsSave = false;
    const migrated = parsed.map((u) => {
      if ("password" in u && u.password !== undefined) {
        const { password: _drop, ...rest } = u as StoredUser & { password: string };
        needsSave = true;
        return { ...rest, passwordHash: MIGRATION_SENTINEL } as StoredUser;
      }
      return u as StoredUser;
    });
    if (needsSave) {
      try { localStorage.setItem(USERS_KEY, JSON.stringify(migrated)); } catch { /* quota */ }
    }
    return migrated;
  } catch {
    return DEFAULT_USERS;
  }
}

export function saveUsers(users: StoredUser[]) {
  try {
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
  } catch { /* quota exceeded — ignore */ }
}

/**
 * Verify a login attempt.
 * Handles the migration path: if the stored hash is the sentinel, the only accepted
 * password is DEMO_PASSWORD; on success the hash is computed and persisted so future
 * logins use PBKDF2 directly.
 * Returns the matching active user or null.
 */
export async function verifyLogin(
  email: string,
  password: string
): Promise<StoredUser | null> {
  const users = getUsers();
  const user = users.find(
    (u) => u.email.toLowerCase() === email.toLowerCase() && u.active
  );
  if (!user) return null;

  if (user.passwordHash === MIGRATION_SENTINEL) {
    // Demo accounts only accept DEMO_PASSWORD during migration.
    if (password !== DEMO_PASSWORD) return null;
    const hash = await hashPassword(password, user.id);
    saveUsers(users.map((u) =>
      u.id === user.id ? { ...u, passwordHash: hash } : u
    ));
    return { ...user, passwordHash: hash };
  }

  const hash = await hashPassword(password, user.id);
  return hash === user.passwordHash ? user : null;
}

/**
 * Hash a new password and persist it for the given user id.
 * Use this when creating or changing a password.
 */
export async function setUserPassword(userId: string, newPassword: string): Promise<void> {
  const users = getUsers();
  const hash = await hashPassword(newPassword, userId);
  saveUsers(users.map((u) => (u.id === userId ? { ...u, passwordHash: hash } : u)));
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
  try {
    const session: Session = {
      userId: user.id,
      name: user.name,
      email: user.email,
      role: user.role,
      loggedInAt: new Date().toISOString(),
    };
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    const existing = JSON.parse(localStorage.getItem("intellcode_settings") ?? "{}");
    localStorage.setItem(
      "intellcode_settings",
      JSON.stringify({ ...existing, name: user.name, email: user.email, role: user.role })
    );
  } catch { /* quota exceeded — ignore */ }
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
  try {
    const raw = localStorage.getItem("intellcode_settings");
    if (raw) {
      const settings = JSON.parse(raw);
      delete settings.name;
      delete settings.email;
      delete settings.role;
      localStorage.setItem("intellcode_settings", JSON.stringify(settings));
    }
  } catch { /* ignore */ }
}

export function generatePassword(): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$";
  const arr = new Uint8Array(12);
  crypto.getRandomValues(arr);
  return Array.from(arr)
    .map((b) => chars[b % chars.length])
    .join("");
}
