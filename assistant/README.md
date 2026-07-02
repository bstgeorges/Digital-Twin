# Digital Twin Assistant — Prototype

Working implementation of the personalized AI assistant designed in the independent study binder (*Designing a Personalized AI Assistant*). Covers **Phases 2–3 and part of 5** of the implementation plan: local server + interface, memory database, and personalization evaluation.

## How it maps to the binder

| Binder section | Implementation |
|---|---|
| 10.2 Interface layer | `index.html` — chat, memory approval panel, activity log |
| 10.3 Language model layer | Hosted Claude API (hybrid design per §12.6) |
| 10.4 Memory layer | `memory.py` — local SQLite, semantic search with keyword fallback |
| 10.6 Safety & logging | Private-memory gate, approval workflow, `actions.jsonl` |
| 9.5 Human-in-the-loop | Extracted memories land as *proposed*; nothing is used until approved |
| 12.5 Data storage rules | Sensitivity classification; `private` memories never sent to the API unless the toggle is on for that request; every memory records its source; inspect/edit/delete supported |
| 13 Evaluation | `eval.py` — blind baseline-vs-personalized comparison report |

## Setup

```bash
cd assistant
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# optional, better retrieval: pip install sentence-transformers

export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## Using it

1. Pick your user (ben / jonathan) — memories are kept separate per user.
2. Seed a few memories manually (style, preferences, current projects).
3. Chat. The assistant retrieves relevant approved memories; the caption under each reply shows which were used.
4. After each exchange the system may **propose** new memories — approve or reject them in the sidebar. Nothing is remembered without approval.
5. Memories marked `private` are stored locally but excluded from API calls unless you check "include private memories".

## Evaluation (binder §13)

```bash
python eval.py --user ben
```

Produces a markdown report with baseline vs. memory-aware responses for blind rating.

## Ethical boundaries implemented (binder §3.5)

No hidden collection (all memories visible + sourced), no auto-approval, user deletion, private-data gate, impersonation disclaimer in the system prompt, uncertainty disclosure instruction.

## Next phases

- **Phase 4 (tools):** start with read-only local file/notes access behind a permission prompt.
- **Adapter fine-tuning (§9.4):** only after the eval loop shows memory-based personalization has plateaued.
