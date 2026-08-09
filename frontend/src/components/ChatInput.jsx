import { useState } from "react";

export default function ChatInput({ onSubmit, disabled }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-end gap-2.5 rounded-xl border border-ink-600 bg-ink-800 p-2.5 focus-within:border-brass/50"
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            handleSubmit(e);
          }
        }}
        rows={1}
        placeholder="Ask about a section of the Penal Code, the Constitution, PECA…"
        className="max-h-32 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-parchment placeholder:text-parchment/35 focus:outline-none"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg bg-emerald px-3.5 text-sm font-medium text-parchment transition-colors hover:bg-emerald-light disabled:cursor-not-allowed disabled:bg-ink-600 disabled:text-parchment/40"
      >
        {disabled ? "Asking…" : "Ask"}
      </button>
    </form>
  );
}
