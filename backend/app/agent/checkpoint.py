"""
Checkpointer for the LangGraph agent.

This is the piece that turns "conversation memory" from a Python variable
(dies on restart, shared across requests unsafely) into real, per-thread,
persistent state.

How it plugs into the existing agent:
    create_react_agent's compiled graph already has a `messages` key in its
    state, using LangGraph's `add_messages` reducer -- every graph step
    appends to that list instead of replacing it. A checkpointer is what
    LangGraph uses to (a) load that list for a given thread_id BEFORE a run
    starts, and (b) persist the updated list after EVERY node the graph
    passes through (agent node, tool node, agent node again, ...), not just
    at the very end. That's what makes memory survive across multiple
    nodes/tools within one turn, and across turns, and across the same
    thread_id, and across an app restart.

Why SQLite specifically: LangGraph also ships MemorySaver (an in-process
dict). MemorySaver would satisfy "remembers previous turns" but NOT
"survives an app restart" -- it would be fake persistence. SqliteSaver
writes every checkpoint to a file, so `thread_id` -> conversation survives
the process dying and starting again, which is the actual requirement.

Set AGENT_MEMORY_DB in .env to change the file location.
"""

import logging
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings

logger = logging.getLogger("agent.checkpoint")

# Tables used by langgraph-checkpoint-sqlite across the versions compatible
# with langgraph==0.2.62. Newer releases split blobs into their own table;
# we defensively try all of them in clear_thread() so this keeps working
# whichever schema version resolves at `pip install` time.
_CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "writes")


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Build (once per process) the checkpointer the agent runs against.

    Cached with lru_cache for the same reason get_llm()/get_vectorstore()
    are elsewhere in this codebase: opening a new SQLite connection per
    request would be wasteful and, worse, would let each request silently
    initialize its own separate connection object.
    """
    db_path = settings.AGENT_MEMORY_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: FastAPI/Streamlit may call this from
    # different threads than the one that opened the connection. We're not
    # doing concurrent writes to the exact same thread_id in practice, so
    # this is the standard, safe way to share one SQLite connection here.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    logger.info("checkpointer_ready db_path=%s", db_path)
    return saver


def clear_thread(thread_id: str) -> None:
    """Delete every persisted checkpoint for exactly one thread_id.

    Every row in every checkpoint table is keyed by thread_id, so a
    `WHERE thread_id = ?` delete only ever touches that one conversation --
    every other session's history is untouched. This is what backs the
    "reset this conversation" feature without wiping other users' threads.
    """
    checkpointer = get_checkpointer()
    conn = checkpointer.conn

    deleted_any = False
    with conn:
        for table in _CHECKPOINT_TABLES:
            try:
                cur = conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                if cur.rowcount:
                    deleted_any = True
            except sqlite3.OperationalError:
                # Table doesn't exist in this schema version -- fine, we
                # tried every known table name on purpose.
                continue

    logger.info("thread_id=%s cleared rows_deleted=%s", thread_id, deleted_any)