"""Evaluation harness — binder section 13 (13.1 style similarity, 13.3 memory recall).

Runs each test prompt twice — WITHOUT memories (generic baseline) and WITH
retrieved memories — and writes a side-by-side markdown report for blind
comparison. Rate each pair yourself; that human judgment is the metric
(binder 13.1: "compare an unpersonalized output with a memory-aware output").

Usage:  ANTHROPIC_API_KEY=... python eval.py --user ben
"""

import argparse
import time
from pathlib import Path

import anthropic

import memory
from app import CHAT_MODEL, SYSTEM_TEMPLATE

TEST_PROMPTS = [
    "Draft a short email asking a teacher for an extension on a project.",
    "Summarize what my current main project is and what's left to do.",
    "Explain how retrieval-augmented generation works.",
    "Help me plan my week.",
]


def run(user: str):
    client = anthropic.Anthropic()
    rows = []
    for prompt in TEST_PROMPTS:
        retrieved = memory.retrieve(user, prompt, k=6)
        mem_text = "\n".join(f"- [{m['type']}] {m['content']}" for m in retrieved) \
                   or "(none)"

        def ask(system):
            r = client.messages.create(model=CHAT_MODEL, max_tokens=800,
                                       system=system,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text

        baseline = ask("You are a helpful assistant.")
        personalized = ask(SYSTEM_TEMPLATE.format(user=user, memories=mem_text))
        rows.append((prompt, mem_text, baseline, personalized))
        print(f"done: {prompt[:50]}")

    out = Path(__file__).parent / f"eval_report_{user}_{int(time.time())}.md"
    with open(out, "w") as f:
        f.write(f"# Evaluation Report — {user}\n\n"
                "For each prompt, rate which response is more useful/personal "
                "(binder 13.1/13.3). B = baseline, P = personalized.\n\n")
        for i, (p, mems, b, pz) in enumerate(rows, 1):
            f.write(f"## {i}. {p}\n\n**Memories retrieved:**\n{mems}\n\n"
                    f"### B (no memory)\n{b}\n\n### P (with memory)\n{pz}\n\n"
                    "**Winner:** ___  **Notes:** ___\n\n---\n\n")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="ben")
    run(ap.parse_args().user)
