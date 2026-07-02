"""Memory layer — binder sections 8 (memory-augmented AI) and 12 (dataset & privacy plan).

Design decisions this file implements:
- Structured memory schema (8.5): type, sensitivity, source, per-user, approval status.
- Human-in-the-loop (9.5): memories arrive as 'proposed' and must be approved before use.
- Privacy (12.5): sensitivity classification; 'private' memories are excluded from
  prompts sent to the hosted API unless the user explicitly enables them per-request.
- Retrieval (8.3/8.4): semantic search via sentence-transformers if installed,
  otherwise a keyword-overlap fallback so the system works with zero heavy deps.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"

MEMORY_TYPES = ("preference", "fact", "project", "style", "constraint")
SENSITIVITIES = ("low", "private")
STATUSES = ("proposed", "approved", "rejected")

# --- optional semantic embeddings -------------------------------------------
_model = None
_tried_model = False


def _get_model():
    global _model, _tried_model
    if not _tried_model:
        _tried_model = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = None  # keyword fallback
    return _model


def _embed(text: str):
    model = _get_model()
    if model is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()


# --- storage -----------------------------------------------------------------

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            sensitivity TEXT NOT NULL DEFAULT 'low',
            status TEXT NOT NULL DEFAULT 'proposed',
            embedding TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    return conn


def add_memory(user: str, type: str, content: str, source: str,
               sensitivity: str = "low", status: str = "proposed") -> dict:
    assert type in MEMORY_TYPES, f"type must be one of {MEMORY_TYPES}"
    assert sensitivity in SENSITIVITIES
    assert status in STATUSES
    now = time.time()
    emb = _embed(content)
    mem = dict(id=str(uuid.uuid4())[:8], user=user, type=type, content=content,
               source=source, sensitivity=sensitivity, status=status,
               created_at=now, updated_at=now)
    with _conn() as c:
        c.execute(
            "INSERT INTO memories VALUES (:id,:user,:type,:content,:source,"
            ":sensitivity,:status,:embedding,:created_at,:updated_at)",
            {**mem, "embedding": json.dumps(emb) if emb else None},
        )
    return mem


def list_memories(user: str | None = None, status: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM memories WHERE 1=1", []
    if user:
        q += " AND user=?"; args.append(user)
    if status:
        q += " AND status=?"; args.append(status)
    q += " ORDER BY created_at DESC"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def set_status(mem_id: str, status: str) -> bool:
    assert status in STATUSES
    with _conn() as c:
        cur = c.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?",
                        (status, time.time(), mem_id))
        return cur.rowcount > 0


def update_content(mem_id: str, content: str) -> bool:
    emb = _embed(content)
    with _conn() as c:
        cur = c.execute(
            "UPDATE memories SET content=?, embedding=?, updated_at=? WHERE id=?",
            (content, json.dumps(emb) if emb else None, time.time(), mem_id))
        return cur.rowcount > 0


def delete_memory(mem_id: str) -> bool:
    """Binder 3.5: user control over memory deletion."""
    with _conn() as c:
        return c.execute("DELETE FROM memories WHERE id=?", (mem_id,)).rowcount > 0


# --- retrieval ---------------------------------------------------------------

_word = re.compile(r"[a-zA-Z0-9]+")


def _keyword_score(query: str, text: str) -> float:
    qw = set(w.lower() for w in _word.findall(query))
    tw = set(w.lower() for w in _word.findall(text))
    if not qw or not tw:
        return 0.0
    return len(qw & tw) / len(qw)


def retrieve(user: str, query: str, k: int = 6,
             include_private: bool = False) -> list[dict]:
    """Return top-k APPROVED memories relevant to the query.

    Permission gate (binder 12.5): private memories are only eligible when the
    user has explicitly enabled them for this request.
    """
    mems = list_memories(user=user, status="approved")
    if not include_private:
        mems = [m for m in mems if m["sensitivity"] != "private"]
    if not mems:
        return []

    q_emb = _embed(query)
    scored = []
    for m in mems:
        if q_emb is not None and m["embedding"]:
            e = json.loads(m["embedding"])
            score = sum(a * b for a, b in zip(q_emb, e))
        else:
            score = _keyword_score(query, m["content"])
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Always include preferences/style even if not query-relevant (binder 7.2/7.3),
    # then top scoring context memories.
    always = [m for s, m in scored if m["type"] in ("preference", "style", "constraint")][:4]
    contextual = [m for s, m in scored
                  if m not in always and s > 0.05][: max(0, k - len(always))]
    return always + contextual
