"""
agent.py - A tool-calling AI agent in pure Python, running on a local Ollama model.

The loop:  ask model -> if it requests tools, run them, feed results back -> repeat
           until the model answers in plain text (or we hit MAX_STEPS).

Usage:  python agent.py                 (interactive)
        python agent.py "What is 847 * 293?"
Needs:  ollama pull llama3.2    (and nomic-embed-text if you use search_pdf)
Tutorial: https://www.techprofree.com/build-ai-agent-python-ollama/
"""

import sys
import ollama
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL = "llama3.2"      # also try: qwen3:4b, gemma3:4b, llama3.1:8b
MAX_STEPS = 6           # guard-rail: stop runaway loops

SYSTEM = """You are a helpful assistant with tools.
Use the calculate tool for any arithmetic - never do math in your head.
Use search_pdf for questions about the document, and read_file for text files.
When you have what you need, answer the user in plain text."""


def run_agent(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response.message

        # No tool calls -> this is the final answer
        if not msg.tool_calls:
            return msg.content

        messages.append(msg)   # keep the model's tool request in the history

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments or {}

            if name not in TOOL_FUNCTIONS:          # model invented a tool
                result = f"Error: unknown tool '{name}'. Available: {list(TOOL_FUNCTIONS)}"
            else:
                try:
                    result = TOOL_FUNCTIONS[name](**args)
                except TypeError as e:              # wrong / missing arguments
                    result = f"Error: bad arguments for {name}: {e}"

            if verbose:
                print(f"  [step {step}] {name}({args}) -> {str(result)[:80]!r}")

            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Stopped: reached the step limit without a final answer."


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(run_agent(" ".join(sys.argv[1:])))
        sys.exit()

    print(f"Agent ready ({MODEL}). Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print("Agent:", run_agent(q), "\n")
