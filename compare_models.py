"""
compare_models.py - Which small Ollama models can actually call tools?

Runs the same five tasks against each model and records whether it picked the
right tool. Results print as a Markdown table you can paste into a blog post.

Usage:  python compare_models.py
Needs:  ollama pull llama3.2 qwen3:4b gemma3:4b   (edit MODELS below)
Tutorial: https://www.techprofree.com/build-ai-agent-python-ollama/
"""

import time
import ollama
import agent
from tools import TOOL_SCHEMAS

MODELS = ["llama3.2", "qwen3:4b", "gemma3:4b"]

# (question, tool we expect it to call first; None = should answer directly)
TASKS = [
    ("What is 847 times 293?", "calculate"),
    ("What is (1234.5 + 678) / 7?", "calculate"),
    ("Read notes.txt and tell me what it says.", "read_file"),
    ("According to the PDF, what is the main topic?", "search_pdf"),
    ("Who wrote Romeo and Juliet?", None),
]


def first_tool(model: str, question: str):
    """Return the name of the first tool the model calls, or None."""
    messages = [{"role": "system", "content": agent.SYSTEM},
                {"role": "user", "content": question}]
    r = ollama.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
    calls = r.message.tool_calls
    return calls[0].function.name if calls else None


print("| Model | " + " | ".join(f"T{i+1}" for i in range(len(TASKS))) + " | Score | Avg time |")
print("|---" * (len(TASKS) + 3) + "|")

for model in MODELS:
    marks, times = [], []
    for question, expected in TASKS:
        t0 = time.time()
        try:
            got = first_tool(model, question)
            ok = (got == expected)
        except Exception as e:
            got, ok = f"ERR {e}", False
        times.append(time.time() - t0)
        marks.append("✅" if ok else f"❌ {got}")
    score = sum(m == "✅" for m in marks)
    print(f"| {model} | " + " | ".join(marks) + f" | {score}/{len(TASKS)} | {sum(times)/len(times):.1f}s |")

print("\nT1-T2 calculator, T3 read_file, T4 search_pdf, T5 should answer with no tool.")
