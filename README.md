# AI Agent in Pure Python with Ollama

A tool-calling AI agent in ~70 lines of plain Python, running on a free local model. No LangChain, no API key. Uses Ollama's native function calling (the model returns structured tool calls — no regex parsing).

📖 **Full tutorial — including which small models can actually call tools, and how the agent fails:**
https://www.techprofree.com/build-ai-agent-python-ollama/

## What's inside

| File | What it does |
|------|--------------|
| `agent.py` | The agent loop: ask → run tools → feed back → repeat, with a step limit |
| `tools.py` | Three real tools: a safe calculator (no `eval`), a sandboxed file reader, and PDF search (RAG) |
| `compare_models.py` | Runs five tasks against several models and prints a tool-calling scorecard |
| `docs/` | Files the agent is allowed to read. Drop a PDF here to enable `search_pdf` |

## Quick start

```bash
ollama pull llama3.2            # the agent's brain (3B, fits in 8 GB RAM)
ollama pull nomic-embed-text    # only needed for search_pdf
pip install -r requirements.txt
python agent.py "What is 847 times 293?"
python agent.py                 # interactive mode
```

## Which model should I use?

Run `python compare_models.py` — it prints a table of which models called the right tool on each task. Results depend on your hardware and model versions, so measure rather than trust the internet.

## License

MIT
