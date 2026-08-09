export default function Header({ productName, tagline, status }) {
  return (
    <header className="border-b border-ink-600 bg-ink-800/60 backdrop-blur px-6 py-5 sm:px-10">
      <div className="mx-auto flex max-w-4xl items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <Seal />
            <h1 className="font-display text-2xl font-semibold text-parchment sm:text-[28px]">
              {productName}
            </h1>
          </div>
          <p className="mt-1.5 pl-[38px] text-sm text-parchment/60">{tagline}</p>
        </div>

        <StatusPill status={status} />
      </div>
    </header>
  );
}

function Seal() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <circle cx="14" cy="14" r="13" stroke="#B08D57" strokeWidth="1.4" />
      <circle cx="14" cy="14" r="9" stroke="#B08D57" strokeWidth="1" opacity="0.6" />
      <path
        d="M9 15.5L12.2 18.5L19 11"
        stroke="#B08D57"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StatusPill({ status }) {
  const map = {
    connecting: { label: "Connecting…", dot: "bg-parchment/40" },
    ready: { label: "Index ready", dot: "bg-emerald-light" },
    error: { label: "Backend unreachable", dot: "bg-rose-400" },
  };
  const { label, dot } = map[status] || map.connecting;

  return (
    <div className="mt-1 flex shrink-0 items-center gap-2 rounded-full border border-ink-600 bg-ink-700 px-3 py-1.5 text-xs text-parchment/70">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="font-mono">{label}</span>
    </div>
  );
}
