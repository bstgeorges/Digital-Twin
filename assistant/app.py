"""Personal AI assistant server — implements binder Phases 2-3.

Layers (binder section 10):
- Interface layer:   web UI served at /
- Language model:    hosted Claude API (hybrid design, binder 12.6)
- Memory layer:      memory.py (local SQLite — sensitive data stays on-device)
- Safety layer:      private-memory gate + human-in-the-loop approval + logging

Run:  ANTHROPIC_API_KEY=sk-... uvicorn app:app --reload
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import memory

app = FastAPI(title="Digital Twin Assistant")

LOG_PATH = Path(__file__).parent / "actions.jsonl"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-5")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "claude-haiku-4-5-20251001")
USERS = ["ben", "jonathan"]


def log(action: str, **details):
    """Binder 10.6: minimal, readable, removable logs."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"ts": time.time(), "action": action, **details}) + "\n")


def _client():
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


# --- chat ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    user: str
    message: str
    include_private: bool = False
    history: list[dict] = []  # [{role, content}]


SYSTEM_TEMPLATE = """You are a personalized assistant for {user}. You approximate \
selected patterns of their communication style, preferences, and project context — \
you are NOT the user and must never claim to be them (impersonation boundary).

Approved memories about {user} (retrieved from their local memory store):
{memories}

Rules:
- Ground answers in the memories above when relevant; say when you are inferring \
rather than recalling (uncertainty disclosure).
- Never invent memories. If you lack context, say so.
- Match the user's stored style/preferences where they exist."""


@app.post("/chat")
def chat(req: ChatRequest):
    retrieved = memory.retrieve(req.user, req.message, k=6,
                                include_private=req.include_private)
    mem_text = "\n".join(
        f"- [{m['type']}/{m['sensitivity']}] {m['content']}" for m in retrieved
    ) or "(no relevant memories stored yet)"

    system = SYSTEM_TEMPLATE.format(user=req.user, memories=mem_text)
    messages = req.history[-10:] + [{"role": "user", "content": req.message}]

    try:
        resp = _client().messages.create(
            model=CHAT_MODEL, max_tokens=1500, system=system, messages=messages)
        answer = resp.content[0].text
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})

    log("chat", user=req.user, message=req.message[:200],
        memories_used=[m["id"] for m in retrieved],
        private_included=req.include_private)

    proposals = _extract_memories(req.user, req.message, answer)
    return {"answer": answer,
            "memories_used": retrieved,
            "proposed_memories": proposals}


EXTRACT_PROMPT = """From this exchange, extract 0-3 durable facts worth remembering \
about the user (preferences, project facts, style, constraints). Skip small talk and \
one-off details. Mark anything personal/sensitive as "private".

User ({user}): {message}
Assistant: {answer}

Reply with ONLY a JSON array (possibly empty):
[{{"type": "preference|fact|project|style|constraint", "content": "...", "sensitivity": "low|private"}}]"""


def _extract_memories(user: str, message: str, answer: str) -> list[dict]:
    """Propose memories for human approval (binder 9.5). Never auto-approved."""
    try:
        resp = _client().messages.create(
            model=EXTRACT_MODEL, max_tokens=500,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(
                user=user, message=message[:2000], answer=answer[:2000])}])
        text = resp.content[0].text.strip()
        start, end = text.find("["), text.rfind("]") + 1
        items = json.loads(text[start:end]) if start != -1 else []
    except Exception:
        return []

    proposals = []
    for it in items[:3]:
        if it.get("type") in memory.MEMORY_TYPES and it.get("content"):
            mem = memory.add_memory(
                user=user, type=it["type"], content=it["content"][:500],
                source="extracted-from-chat",
                sensitivity=it.get("sensitivity", "low"),
                status="proposed")
            proposals.append(mem)
            log("memory_proposed", id=mem["id"], user=user)
    return proposals


# --- memory management ----------------------------------------------------------

class MemoryIn(BaseModel):
    user: str
    type: str
    content: str
    sensitivity: str = "low"


@app.get("/memories")
def get_memories(user: str | None = None, status: str | None = None):
    return memory.list_memories(user=user, status=status)


@app.post("/memories")
def create_memory(m: MemoryIn):
    # Manually entered memories are trusted -> approved immediately (binder 12.2)
    mem = memory.add_memory(user=m.user, type=m.type, content=m.content,
                            source="manual", sensitivity=m.sensitivity,
                            status="approved")
    log("memory_added", id=mem["id"], user=m.user, source="manual")
    return mem


@app.post("/memories/{mem_id}/approve")
def approve(mem_id: str):
    ok = memory.set_status(mem_id, "approved")
    log("memory_approved", id=mem_id)
    return {"ok": ok}


@app.post("/memories/{mem_id}/reject")
def reject(mem_id: str):
    ok = memory.set_status(mem_id, "rejected")
    log("memory_rejected", id=mem_id)
    return {"ok": ok}


@app.delete("/memories/{mem_id}")
def remove(mem_id: str):
    ok = memory.delete_memory(mem_id)
    log("memory_deleted", id=mem_id)
    return {"ok": ok}


# --- logs & meta -----------------------------------------------------------------

@app.get("/logs")
def get_logs(limit: int = 100):
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()[-limit:]
    return [json.loads(l) for l in reversed(lines)]


@app.get("/users")
def get_users():
    return USERS


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
