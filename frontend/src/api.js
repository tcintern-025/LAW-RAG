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
