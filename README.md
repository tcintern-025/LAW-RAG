# Multi-Step LangGraph Agent — Expansion Notes

This document covers the upgrade from a basic tool-calling agent to a
hand-built, multi-step LangGraph `StateGraph`. It's a companion to the main
[`README.md`](./README.md), which still covers the base RAG system, setup,
and deployment.

## 1. Project overview

The base project is the **Pakistan Law RAG Assistant**: a grounded,
source-cited Q&A system (LangChain + ChromaDB + Hugging Face embeddings +
Groq) that answers legal questions strictly from indexed documents
(Constitution, Penal Code, Contract Act, PECA), refusing to guess when the
documents don't cover a question.

On top of that RAG core sits an **agent layer** — a tool-calling assistant
that can decide *whether* a question needs document search, arithmetic,
today's date, or nothing at all, instead of always retrieving.

## 2. What was added

**Before:** `app/agent/graph.py` called
`langgraph.prebuilt.create_react_agent(llm, TOOLS)` — a library helper that
handles the tool-calling loop internally. It works, but the State schema,
nodes, edges, and routing logic are hidden inside the library, and each call
to `run_agent()` started from a blank slate with no memory of earlier turns.

**After:** the same file now builds an explicit `StateGraph` by hand —
a typed `AgentState`, two nodes (`agent_node`, `tool_node`), a conditional
edge that decides whether to call a tool or finish, an edge that loops tool
results back to the agent, and a `MemorySaver` checkpointer that gives the
agent real conversation memory across turns.

```text
Basic AI Agent                          Multi-Step LangGraph Agent
───────────────                         ──────────────────────────
create_react_agent() black box    →     Hand-built StateGraph: explicit
                                         State, nodes, edges, routing

One tool call, then stop          →     Genuine multi-step loop: tool
(loop existed but was implicit,          results feed back into the agent,
not inspectable/testable)                which can call a SECOND tool
                                         before answering

No memory between requests        →     MemorySaver checkpointer keyed by
                                         thread_id — follow-up questions
                                         ("explain that simply") resolve
                                         against the prior turn

tools_used only                   →     tools_used + execution_trace, a
                                         step-by-step path through the graph

Tool errors caught only inside    →     Two layers: each tool still catches
each tool function                       its own errors, AND tool_node has
                                         a safety net for anything a tool
                                         doesn't catch itself

No tracing hooks                  →     Optional LangSmith tracing via
                                         standard env vars, zero code changes
```

Nothing about the RAG chain, retriever, vectorstore, ingestion pipeline, or
the three tools' internal logic changed. This was scoped entirely to the
agent orchestration layer plus the API/UI surface needed to expose the new
capabilities (thread_id, execution trace).

## 3. Architecture

### State (`app/agent/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # accumulates
    tools_used: list[str]                                  # last tool_node batch
    error: str | None                                      # last tool failure, if any
```

`messages` uses LangGraph's `add_messages` reducer, so every node's return
value is *appended* to the running list rather than replacing it. That one
design choice is what makes both multi-step tool loops (within a turn) and
multi-turn conversation memory (across turns, via the checkpointer) work
without any manual list-merging code.

### Nodes

| Node | File | Responsibility |
|---|---|---|
| `agent_node` | `graph.py` | Calls the Groq LLM (bound to all 3 tools) with the full message history. Returns either a plain-text answer or one/more tool calls. |
| `tool_node` | `graph.py` | Executes whatever tool(s) the last AI message requested. Wraps each call in `try/except`, always producing a `ToolMessage` — even on failure — so the graph never crashes. |

### Edges & conditional routing

```text
START ──► agent_node
              │
       route_after_agent()
        /            \
  tool_calls?      no tool_calls?
       │                │
       ▼                ▼
   tool_node           END
       │
       └──────► agent_node   (always loops back)
```

```python
def route_after_agent(state) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"
```

The loop-back edge (`tool_node → agent_node`, unconditional) combined with
this conditional edge is what makes the path length genuinely variable: 0
tool calls (direct answer), 1 tool call, or 2+ tool calls in sequence — the
graph doesn't hard-code any of these, the model's own output decides.

### Tool result handling

`tool_node` appends a `ToolMessage` per tool call to `state["messages"]`.
Because of the `add_messages` reducer, when `agent_node` runs again it sees
the *entire* prior conversation, including that tool result, and decides
its next move with that context — not just the original question.

### Multi-step execution (concrete example)

```text
User: "Find the fine for theft, then triple it."
  ↓
agent_node   → tool_call: search_legal_documents("theft fine amount")
  ↓
tool_node    → ToolMessage: "...Section 379...fine..."
  ↓
agent_node   → tool_call: calculate("50000 * 3")     ← sees the retrieved fine
  ↓
tool_node    → ToolMessage: "150000"
  ↓
agent_node   → final answer, no more tool_calls
  ↓
END
```

### Error handling

Two layers, deliberately redundant:

1. Each tool in `app/agent/tools.py` already wraps its body in
   `try/except` and returns a readable string like
   `"calculate tool failed: could not evaluate '50000/0' (division by zero)"`.
2. `tool_node` in `graph.py` *also* wraps the call in `try/except`, catching
   anything a tool doesn't catch itself (e.g. an unexpected `RuntimeError`
   from `answer_question()` if `GROQ_API_KEY` is missing). Either way, the
   result is always a `ToolMessage` fed back to the agent — never an
   unhandled exception reaching the API layer. `state["error"]` records the
   most recent failure so the API/UI can flag a degraded answer.
3. `run_agent()` itself wraps `graph.invoke()` in `try/except` as a final
   safety net and re-raises as a clean `RuntimeError` the FastAPI layer
   already knows how to turn into a 500 with a readable `detail`.

### Conversation history

```python
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# turn 1
run_agent("What does Article 25 say?", thread_id="abc")
# turn 2 — same thread_id, agent sees turn 1's messages automatically
run_agent("Explain that in simple words.", thread_id="abc")
```

`thread_id` is optional. Omit it (or pass `None`) for a stateless, one-off
question — `run_agent()` generates a fresh UUID so the call still works, it
just has no history to draw on. The FastAPI endpoint and the Streamlit app
both handle this: the API echoes back whatever `thread_id` was used so the
caller can continue the same thread; the Streamlit app keeps one
`thread_id` per browser session (`st.session_state.agent_thread_id`), with
a sidebar button to reset it.

`MemorySaver` is in-process memory — fine for a single-server demo/deploy.
Swap it for LangGraph's `SqliteSaver` (or a Postgres-backed checkpointer) if
history needs to survive a process restart; no other code changes needed.

## 4. Tools

| Tool | Purpose | Input | Output | When the agent uses it |
|---|---|---|---|---|
| `search_legal_documents` | Searches the indexed legal documents via the existing RAG chain (`answer_question()`) | `query: str` | Grounded answer + source list, as one string | Any question about Pakistani law, rights, penalties, or specific sections |
| `calculate` | AST-based safe arithmetic (no `eval()`) | `expression: str`, e.g. `"50000 * 3"` | Numeric result as a string, or a readable failure message | Totaling fines, splitting compensation, converting sentence lengths — especially after a `search_legal_documents` call surfaces a number |
| `get_current_date` | Returns today's UTC date | none | `"YYYY-MM-DD (UTC)"` | Computing legal deadlines (e.g. the 24-hour production-before-magistrate rule under Article 10) |

All three were already implemented and unchanged — this expansion only
changed *how* they're orchestrated, not what they do.

## 5. Example execution

```text
User: "What's the fine for theft under Section 379, and what would three
       such fines add up to?"
  ↓
agent_node   decides: needs to look up the law first
  ↓
tool: search_legal_documents("theft fine Section 379")
  ↓
State updated: ToolMessage with retrieved section text + sources
  ↓
agent_node   reads retrieved context, decides: now needs arithmetic
  ↓
tool: calculate("<retrieved amount> * 3")
  ↓
State updated: ToolMessage with the numeric result
  ↓
agent_node   composes final answer citing the section AND the total
  ↓
END — execution_trace: ["AGENT", "search_legal_documents", "AGENT", "calculate", "AGENT"]
```

## 6. Error handling in practice

If `calculate` is asked to evaluate something invalid (division by zero, a
malformed expression, or an attempted injection like `__import__('os')`),
the AST-based evaluator rejects it, the tool returns a string describing
the failure, `tool_node` records it in `state["error"]`, and `agent_node`
sees that failure message and explains the limitation to the user in plain
language on its next turn — the graph does not crash and does not silently
return a wrong answer.

## 7. LangSmith

Not required — the agent works identically with tracing off. To enable it:

```bash
# in backend/.env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_from_smith.langchain.com
LANGCHAIN_PROJECT=pakistan-law-assistant
```

LangChain/LangGraph read these environment variables automatically — no
code changes. With tracing on, each run in [smith.langchain.com](https://smith.langchain.com)
shows: the user's input, every `agent_node` LLM call (with its reasoning
and any tool calls it emitted), every `tool_node` execution and its result,
each state transition, the final response, and any errors — nested exactly
along the graph's actual execution path.

## 8. Before vs after

**Before:** a single-shot-or-loop agent (`create_react_agent`) that could
call tools but had no inspectable architecture, no persisted memory across
API calls, and surfaced only a flat `tools_used` list with no sense of
*order* or *why*.

**After:** an explicit, testable `StateGraph` with a typed state schema,
two clearly separated nodes, conditional routing that's a plain Python
function you can unit-test, genuine multi-step tool chaining driven by the
model's own decisions, two-layer error handling, conversation memory keyed
by `thread_id`, and a step-by-step execution trace exposed all the way to
the API response and the Streamlit UI.

## 9. Key learning

The core idea a multi-step agent demonstrates is that **"agent" means the
control flow is decided at runtime by the model's output, not hard-coded by
the developer.** A single `if/else` that calls one tool and returns isn't
an agent — it's a router. What makes this genuinely agentic is the
loop-back edge (`tool_node → agent_node`) combined with conditional
routing: every time control returns to the agent, it re-evaluates the
*entire* accumulated state and can choose a different path — call another
tool, call the same tool again with different arguments, or stop — based on
what it has learned so far. State is what carries that "what it has learned
so far" between decisions; without a properly-reducing state schema
(`add_messages`), that memory silently breaks.

---

## Testing

The graph's mechanics (state accumulation, conditional routing, the
tool→agent loop, error handling, and thread-scoped memory) were verified
with a scripted fake LLM standing in for Groq — this sandbox can't reach
`api.groq.com`, so this substitutes a `ScriptedLLM.invoke()` that returns
canned `AIMessage`/tool-call objects in place of `ChatGroq`, while running
through the real `app.agent.graph` module unmodified otherwise.

| # | Test | Input | Expected | Result |
|---|---|---|---|---|
| 1 | Direct answer, no tool | "hi there" | `tools_used=[]`, trace=`["AGENT"]` | ✅ PASS |
| 2 | Tool 1 only | "What is the punishment for theft?" | `tools_used=["search_legal_documents"]` | ✅ PASS |
| 3 | Tool 2 only | "What's 50000 + 25000?" | `tools_used=["calculate"]` | ✅ PASS |
| 4 | Multi-step | "Find the fine for theft, then triple it." | `tools_used=["search_legal_documents","calculate"]`, trace has both in order | ✅ PASS |
| 5 | Tool failure | "What is 50000 divided by 0?" | No crash; `state["error"]` set; agent still returns a plain-language answer | ✅ PASS |
| 6 | Conversation history | Turn 1: "What does Article 25 say?" → Turn 2 (same `thread_id`): "Explain that in simple words." | Turn 2's LLM call sees turn 1's answer in its message history | ✅ PASS |
| 6b | Thread isolation | Same turn-2 question, different `thread_id` | No cross-talk — history from thread `t6` is NOT visible to thread `t7-different` | ✅ PASS |

**Not exercised in this sandbox** (no network access to `api.groq.com` or
`huggingface.co` from this environment): a live end-to-end run against the
real Groq model and the real ChromaDB-backed `search_legal_documents`. The
graph wiring these tests validate is exactly what the real LLM/tools plug
into unchanged — `_get_llm()` is the only swapped seam — but you should run
`streamlit run streamlit_app.py` (or hit `POST /agent/ask`) locally with a
real `GROQ_API_KEY` before considering this production-verified end to end.

---

## Files changed

| File | Change |
|---|---|
| `backend/app/agent/state.py` | **New.** `AgentState` TypedDict. |
| `backend/app/agent/graph.py` | **Rewritten.** Hand-built `StateGraph` replacing `create_react_agent`. |
| `backend/app/agent/tools.py` | Unchanged. |
| `backend/app/schemas.py` | Added `thread_id` to `AgentAskRequest`; added `execution_trace`, `thread_id`, `error` to `AgentAskResponse`. |
| `backend/app/main.py` | `/agent/ask` now passes `thread_id` through and returns the richer response. |
| `backend/.env.example` | Added optional `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT`. |
| `backend/requirements.txt` | Added `langsmith==0.2.10` (optional, only active if tracing is enabled). |
| `backend/streamlit_app.py` | Agent mode now threads a per-session `thread_id`, shows the execution trace, shows tool errors, and has a "start new agent conversation" reset button. |
| `README_LANGGRAPH.md` | **New** — this file. |

## Git commands

```bash
git add backend/app/agent/state.py backend/app/agent/graph.py \
        backend/app/schemas.py backend/app/main.py \
        backend/.env.example backend/requirements.txt \
        backend/streamlit_app.py README_LANGGRAPH.md

git commit -m "feat: add LangGraph state and agent nodes"
git commit -m "feat: add multi-tool routing workflow" --allow-empty  # or split into logical commits as you go
git commit -m "feat: add tool result handling"
git commit -m "feat: add graceful tool error handling"
git commit -m "feat: add conversation memory via thread_id checkpointer"
git commit -m "docs: add multi-step agent architecture README"

git push origin main
```

(Split into as many real commits as you like as you apply these changes —
the message above is illustrative; make the commits match your actual
diffs rather than one giant commit.)

## Remaining issues / things to verify with real credentials

1. **Live Groq test** — run the 6 test scenarios above against the real
   `ChatGroq` model once `GROQ_API_KEY` is set locally, to confirm the real
   model's tool-selection behavior matches what the scripted fake exercised
   structurally.
2. **LangSmith** — untested end-to-end since it requires a real API key;
   the env-var wiring is standard LangChain behavior, but worth a quick
   sanity check that traces actually appear in your dashboard.
3. **React frontend** — `frontend/src/api.js` / `App.jsx` still only call
   `/ask`, not `/agent/ask`. If you want agent mode in the React UI too
   (not just Streamlit), that's a follow-up, not something this expansion
   touched.
4. **Persistent checkpointer** — `MemorySaver` loses history on process
   restart. Fine for Streamlit Community Cloud's single-process model, but
   worth swapping to `SqliteSaver`/Postgres if you deploy the FastAPI
   backend somewhere with multiple workers or frequent restarts.