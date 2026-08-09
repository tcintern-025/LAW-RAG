import { useState } from "react";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export default function ExhibitList({ sources }) {
  const [openIndex, setOpenIndex] = useState(null);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 border-t border-ink-600 pt-3">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-wider text-brass/80">
        Referenced excerpts
      </p>
      <div className="flex flex-col gap-1.5">
        {sources.map((s, i) => {
          const label = `Exhibit ${LETTERS[i] || i + 1}`;
          const isOpen = openIndex === i;
          return (
            <div
              key={s.chunk_id || i}
              className="rounded-md border border-ink-600 bg-ink-800/70"
            >
              <button
                onClick={() => setOpenIndex(isOpen ? null : i)}
                className="flex w-full items-center gap-3 px-3 py-2 text-left"
                aria-expanded={isOpen}
              >
                <span className="flex h-6 min-w-[2.75rem] items-center justify-center rounded border border-brass/40 bg-brass/10 px-1.5 font-mono text-[10px] font-medium text-brass-light">
                  {label}
                </span>
                <span className="flex-1 truncate text-sm text-parchment/85">
                  {s.source}
                  {s.page != null && (
                    <span className="text-parchment/45"> · p.{s.page + 1}</span>
                  )}
                </span>
                <Chevron open={isOpen} />
              </button>
              {isOpen && (
                <p className="border-t border-ink-600 px-3 py-2.5 font-mono text-[12.5px] leading-relaxed text-parchment/70">
                  {s.excerpt}
                  {s.excerpt.length >= 400 ? "…" : ""}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Chevron({ open }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      className={`shrink-0 text-parchment/40 transition-transform ${open ? "rotate-180" : ""}`}
    >
      <path
        d="M3.5 5.25L7 8.75L10.5 5.25"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
