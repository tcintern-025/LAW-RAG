import { useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import { askQuestion, getHealth } from "./api";

const PRODUCT_NAME = "Pakistan Law Assistant";
const TAGLINE = "Grounded answers, sourced from indexed legal documents.";

const SUGGESTIONS = [
  "What safeguards exist for a person who is arrested?",
  "What is the punishment for theft under the Penal Code?",
  "What makes a contract enforceable?",
  "What counts as cyberstalking under PECA?",
];

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then(() => setStatus("ready"))
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function handleAsk(question) {
    setMessages((m) => [...m, { role: "user", content: question }]);
    setLoading(true);
    try {
      const result = await askQuestion(question);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
          hasSufficientContext: result.has_sufficient_context,
          disclaimer: result.disclaimer,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          error: true,
          content:
            err.message ||
            "Something went wrong reaching the assistant. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full min-h-screen flex-col bg-ink">
      <Header productName={PRODUCT_NAME} tagline={TAGLINE} status={status} />

      <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-4 sm:px-10">
        <div ref={scrollRef} className="flex-1 overflow-y-auto py-8">
          {messages.length === 0 ? (
            <EmptyState onPick={handleAsk} />
          ) : (
            <div className="flex flex-col gap-5">
              {messages.map((m, i) => (
                <ChatMessage key={i} message={m} />
              ))}
              {loading && <ThinkingBubble />}
            </div>
          )}
        </div>

        <div className="sticky bottom-0 bg-ink pb-6 pt-3">
          <ChatInput onSubmit={handleAsk} disabled={loading || status === "error"} />
          <p className="mt-2.5 text-center text-[11px] text-parchment/30">
            Educational demo · Not a substitute for professional legal advice
          </p>
        </div>
      </main>
    </div>
  );
}

function EmptyState({ onPick }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 py-16 text-center">
      <div>
        <p className="font-display text-lg text-parchment/80">
          Ask a question grounded in the indexed documents.
        </p>
        <p className="mt-1.5 text-sm text-parchment/45">
          Every answer names the document and section it came from.
        </p>
      </div>
      <div className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-lg border border-ink-600 bg-ink-800 px-3.5 py-2.5 text-left text-[13.5px] text-parchment/70 transition-colors hover:border-brass/40 hover:text-parchment"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm border border-ink-600 bg-ink-800 px-4 py-3.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-parchment/40"
            style={{ animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
    </div>
  );
}
