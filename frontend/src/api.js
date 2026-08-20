// Base URL of the backend API. Set VITE_API_URL in a .env file (or in your
// hosting provider's environment settings) to point this at the deployed
// backend instead of localhost. See .env.example.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore — use statusText
    }
    throw new Error(detail);
  }

  return res.json();
}

export function askQuestion(question) {
  return request("/ask", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function getHealth() {
  return request("/health");
}

export function getSources() {
  return request("/sources");
}

// --- Agent (memory-aware) -------------------------------------------------
// Pass `threadId` (from a previous askAgent() response) to continue that
// same conversation — the agent will have prior turns available. Omit it
// on the first call of a new conversation; the server mints one and
// returns it in the response, ready to pass into the next call.
export function askAgent(question, threadId) {
  return request("/agent/ask", {
    method: "POST",
    body: JSON.stringify({ question, thread_id: threadId ?? null }),
  });
}

// Clears only the given thread's memory. Every other conversation
// (this user's other tabs, or any other user's threads) is unaffected.
export function resetConversation(threadId) {
  return request("/agent/reset", {
    method: "POST",
    body: JSON.stringify({ thread_id: threadId }),
  });
}