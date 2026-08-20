# Conversation Memory in the Agent

This document explains the memory feature added to the LangGraph agent
(`app/agent/graph.py`). The grounded RAG path (`/ask`, `app/rag/chain.py`)
is unchanged and intentionally remains single-shot/stateless.

## Before

```
User question ──► Agent (fresh every time) ──► Tool (if needed) ──► Answer
```

Every call to `run_agent()` started from an empty message list. Two calls in
a row about "Article 25" and "explain it simply" were unrelated to the
agent — it had no way to know what "it" meant.

## After

```
User question + thread_id
        │
        ▼
POST /agent/ask
        │
        ▼
run_agent(question, thread_id)
        │
        ▼
config = {"configurable": {"thread_id": thread_id}}
        │
        ▼
SqliteSaver loads this thread's prior `messages` ◄──┐
        │                                            │
        ▼                                            │
Agent node (sees full history) ──► Tool node ──► writes checkpoint
        │                                            │
        ▼                                            │
Agent node (sees history + tool result) ─────────────┘
        │
        ▼
Final answer, `messages` persisted to disk under thread_id
```

## Concepts

**State** — `create_react_agent` (from `langgraph.prebuilt`) already builds a
graph whose state has a `messages` key using LangGraph's `add_messages`
reducer. A reducer controls how a node's return value combines with the
existing value for that key; `add_messages` means "append the new
message(s), don't replace the list." This project didn't need a new state
schema — it needed a way to keep that existing list around between calls,
which is exactly what a checkpointer does.

**Message history** — the literal list of `HumanMessage`/`AIMessage`/tool
messages for one conversation, in order. This is what "remembering the
conversation" means concretely: the LLM is shown that whole list (not just
the newest question) on every turn.

**Thread ID** — an arbitrary string that scopes a message history. Every
checkpoint row is stored keyed by `thread_id`. Two different thread IDs are
two completely separate conversations, even against the same SQLite file,
the same process, the same client.

**Session** — in this project, a "session" is just whichever thread_id a
particular caller is currently using — a browser tab's `st.session_state`
value in Streamlit, or whatever the frontend chooses to hold onto and send
back in `AgentAskRequest.thread_id`.

**Checkpoint** — a snapshot of the graph's full state, saved after every
node the graph runs through (not only at the end of a turn). This is what
makes memory survive a multi-node run: if the agent calls a tool and then
continues, the state is already checkpointed by the time the tool result
comes back, not held only in a local Python variable during that gap.

**Persistent memory** — checkpoints written to a SQLite file
(`AGENT_MEMORY_DB` in `.env`, default `backend/data/conversations.sqlite`)
instead of an in-memory dict. Restarting the FastAPI/Streamlit process and
reusing the same `thread_id` resumes the conversation from disk.

**State across nodes** — because the checkpointer, not a Python variable,
is the source of truth, `messages` is fully available to the agent node
both before and after any tool call in the same turn, and to the next turn
entirely.

**Conversation reset** — `reset_conversation(thread_id)` /
`POST /agent/reset` deletes only the rows in the SQLite checkpoint tables
matching that one `thread_id` (`app/agent/checkpoint.py:clear_thread`).
Every other thread's rows are untouched, because every row in every
checkpoint table is keyed by `thread_id`.

## Example — multi-turn conversation

```
User:      What is Article 25?
Assistant: Article 25 concerns equality of citizens before law... [search_legal_documents]

User:      Explain it in simple words.
Assistant: In plain terms, Article 25 says everyone is treated the same
           by the law, and the law can still make special rules to help
           protect women and children. [no tool needed — already has the
           text from the previous turn]

User:      Does it apply to all citizens?
Assistant: Yes — Article 25 states all citizens are equal before the law
           and entitled to equal protection of law... [search_legal_documents,
           because the agent re-grounds the specific legal claim even
           though it already knows the topic from context]
```

The second and third turns never repeat "Article 25" — the agent resolves
"it" from the message history the checkpointer loaded.

## Multiple sessions

```
thread: law_001                      thread: law_002
User: What is Article 25?            User: What is PECA?
Assistant: ...                       Assistant: ...
User: Explain it simply.
Assistant: ... (still Article 25)
```

`law_001` and `law_002` are separate rows in SQLite. Asking "what does it
mean" in `law_002` would have nothing to resolve "it" against — because
that thread's history never mentioned Article 25.

## Tool + memory

The agent decides which tool to call using the *whole* message history, not
just the newest message. "Does it apply to all citizens?" on its own has no
legal subject for `search_legal_documents` to search for — with the prior
turn in context, the model reformulates internally and searches for Article
25 specifically. This is the same mechanism (full message history feeding
tool-call decisions), not a special case.

## Reset

```
Old conversation (thread_id = law_001):
User: What is Article 25?
Assistant: ...

POST /agent/reset {"thread_id": "law_001"}
        ↓
Rows deleted WHERE thread_id = 'law_001' only
        ↓
Next message on law_001 starts with zero prior context
        ↓
law_002's rows are completely unaffected
```

## LangSmith tracing

Already wired through: `app/config.py` calls `load_dotenv()`, and LangChain/
LangGraph read `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and
`LANGCHAIN_PROJECT` directly from the environment — no code change needed
beyond what's in `.env.example`.

To enable it:
1. Create a free account and API key at https://smith.langchain.com.
2. In `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   LANGCHAIN_PROJECT=pakistan-law-assistant
   ```
3. Restart the app. Every `run_agent()` call now shows up in the LangSmith
   UI as a trace: the `thread_id` in the run's metadata, the agent node,
   which tool was selected, the tool's result, and the final response —
   inspectable per-turn.

Never commit `LANGCHAIN_API_KEY` — it's already covered by `.gitignore`'s
`.env` entry, same as `GROQ_API_KEY`.

## Testing performed

| # | Test | Result |
|---|------|--------|
| 1 | Basic memory: "What is Article 25?" → "Explain it simply." | Second answer resolves "it" to Article 25 via loaded message history — verified by code path (checkpointer loads prior `messages` before `invoke()`; no re-mention of the article name required in the second question). |
| 2 | 3-turn conversation, same subject throughout | Same mechanism as #1, extended — each turn's `invoke()` receives one new `HumanMessage`, with all prior turns supplied by the checkpointer. |
| 3 | Tool + memory: "What does Article 25 say?" → "Does it apply to all citizens?" | Verified `tools_used` is computed only over `result["messages"][len(prior_messages):]`, so it correctly reports tools invoked in *this* turn even though the model can see — and is influenced by — every prior turn. |
| 4 | Multiple sessions (`law_001` vs `law_002`) don't interfere | Verified structurally: every checkpoint row is keyed by `thread_id`; `get_state(config)` and `invoke(..., config)` always scope to `config["configurable"]["thread_id"]`. |
| 5 | Reset clears only the targeted thread | Verified `clear_thread()`'s `DELETE ... WHERE thread_id = ?` against each known checkpoint table name. |
| 6 | Persistence across restart | Verified structurally: `SqliteSaver` is backed by a real `sqlite3.connect()` file connection (`AGENT_MEMORY_DB`), not `MemorySaver`'s in-process dict — the same `thread_id` against the same file after a restart loads the same rows. |
| 7 | Multi-node state (agent → tool → agent) | Verified structurally: the checkpointer is attached to the *compiled graph* via `create_react_agent(..., checkpointer=...)`, so LangGraph checkpoints after every superstep the graph executes, not only at `invoke()` return. |

> **Note on how these were verified:** this pass didn't have a live
> `GROQ_API_KEY` or the (fairly heavy) `sentence-transformers`/`chromadb`
> stack installed to run an end-to-end call. All new/changed Python files
> passed `python -m py_compile` (syntax-checked). The verifications above
> are structural/code-path reviews against the actual LangGraph
> `create_react_agent` + `SqliteSaver` contract (confirmed against current
> LangGraph docs/source), not a live conversation transcript. **Before you
> trust this in production, run the 7 tests above yourself** with a real
> `GROQ_API_KEY` — that's the one thing I can't verify from here.

## Files changed

| File | Change |
|---|---|
| `app/agent/checkpoint.py` | **New.** `get_checkpointer()`, `clear_thread()`. |
| `app/agent/graph.py` | `get_agent()` now passes a checkpointer; `run_agent()` takes `thread_id`; new `reset_conversation()`. |
| `app/config.py` | +1 setting: `AGENT_MEMORY_DB`. |
| `app/schemas.py` | `AgentAskRequest`/`Response` gain `thread_id`; new `ResetConversationRequest`/`Response`. |
| `app/main.py` | `/agent/ask` generates/echoes `thread_id`; new `/agent/reset`. |
| `streamlit_app.py` | Per-session `thread_id` in `st.session_state`; sidebar "New conversation" / "Reset" controls (Agent mode only). |
| `frontend/src/api.js` | New `askAgent(question, threadId)`, `resetConversation(threadId)` — additive, nothing existing changed. |
| `.env.example` | `AGENT_MEMORY_DB`, LangSmith vars. |
| `requirements.txt` | +`langgraph-checkpoint-sqlite`. |
| `.gitignore` | Excludes `data/conversations.sqlite*`. |

Unchanged, on purpose: `app/rag/chain.py`, `app/retrieval/`, `app/ingestion/`,
`app/agent/tools.py`, the `/ask` endpoint, and the React chat UI's existing
`/ask` wiring.

## What's NOT done / your call

- **Grounded RAG mode (`/ask`) still has no memory.** That's a deliberate
  scope decision, not an oversight — it's a plain function without a
  message-list state, and the assignment centers on the *agent's* memory.
  If you want memory there too, the cleanest path is routing it through the
  same agent graph (it already has a `search_legal_documents` tool that
  wraps this exact chain) rather than building a second memory system.
- **The React frontend doesn't call `/agent/ask` yet at all** (it only
  calls `/ask`). `api.js` now has `askAgent()`/`resetConversation()` ready
  to use, but wiring a thread-aware chat UI into `App.jsx` is a separate,
  sizeable frontend task I didn't do here to avoid touching working UI code
  beyond what was asked.
- **Concurrent writers to the same thread_id** aren't specially guarded
  beyond SQLite's own locking — fine for one agent process; if you ever run
  multiple backend replicas against the same SQLite file, move to
  `langgraph-checkpoint-postgres` instead (same `checkpointer=` API).

## Git

```bash
git add backend/app/agent/checkpoint.py \
        backend/app/agent/graph.py \
        backend/app/config.py \
        backend/app/schemas.py \
        backend/app/main.py \
        backend/streamlit_app.py \
        backend/.env.example \
        backend/.gitignore \
        backend/requirements.txt \
        frontend/src/api.js \
        docs/CONVERSATION_MEMORY.md

git commit -m "feat: add LangGraph conversation state via SqliteSaver checkpointer"
git commit -m "feat: add thread-based conversation memory to the agent"
git commit -m "feat: add persistent checkpointing (SQLite)"
git commit -m "feat: add conversation reset (per-thread only)"
git commit -m "feat: integrate memory with tool calling in run_agent"
git commit -m "docs: document LangGraph conversation memory"

git push
```

(Split into that many commits if you want the granular history the spec
asked for; a single `feat: add conversation memory to the LangGraph agent`
commit is equally fine if you'd rather squash it.)