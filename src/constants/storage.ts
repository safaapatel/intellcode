/** Single source of truth for all localStorage keys used by IntelliCode. */
export const STORAGE_KEYS = {
  reviewHistory:  "intellcode_review_history",
  settings:       "intellcode_settings",
  repositories:   "intellcode_repositories",
  rules:          "intellcode_rules",
  feedback:       "intellcode_feedback",
  onboarding:     "intellcode_onboarding_done",
  session:        "intellcode_session",
  theme:          "intellcode_theme",
} as const;
