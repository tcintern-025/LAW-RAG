import ExhibitList from "./ExhibitList";

export default function ChatMessage({ message }) {
  const { role, content, sources, hasSufficientContext, disclaimer, error } = message;

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-emerald px-4 py-2.5 text-[15px] leading-relaxed text-parchment">
          {content}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-rose-900/50 bg-rose-950/30 px-4 py-3 text-[14.5px] leading-relaxed text-rose-200">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-ink-600 bg-ink-800 px-4 py-3.5 text-[15px] leading-relaxed text-parchment/90">
        {!hasSufficientContext && (
          <div className="mb-2 inline-flex items-center gap-1.5 rounded-full border border-brass/30 bg-brass/10 px-2.5 py-0.5 font-mono text-[10.5px] text-brass-light">
            Insufficient context in indexed documents
          </div>
        )}
        <p className="whitespace-pre-wrap">{content}</p>

        <ExhibitList sources={sources} />

        {disclaimer && (
          <p className="mt-3 border-t border-ink-600 pt-2.5 text-[11.5px] italic leading-relaxed text-parchment/40">
            {disclaimer}
          </p>
        )}
      </div>
    </div>
  );
}
