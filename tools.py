"""
tools.py - The three tools the agent can call, plus their JSON descriptions.

Every tool is a plain Python function that takes strings and returns a string.
Tutorial: https://www.techprofree.com/build-ai-agent-python-ollama/
"""

import ast
import operator
from pathlib import Path

# ---------------------------------------------------------------- 1. calculator
# A real calculator that does NOT use eval(). It walks the expression's syntax
# tree and only allows numbers and these operators - nothing else can run.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / ** % are allowed")

def calculate(expression: str) -> str:
    """Evaluate a math expression such as '847 * 293' or '(12.5 + 3) / 4'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------- 2. file reader
DOCS_DIR = Path(__file__).parent / "docs"   # the agent may only read files in here

def read_file(filename: str) -> str:
    """Return the text of a file inside the docs/ folder (first 4000 characters)."""
    path = (DOCS_DIR / Path(filename).name).resolve()   # .name strips any ../ tricks
    if not path.is_file():
        available = ", ".join(p.name for p in DOCS_DIR.iterdir()) or "none"
        return f"Error: '{filename}' not found. Files available: {available}"
    return path.read_text(encoding="utf-8", errors="ignore")[:4000]


# ---------------------------------------------------------------- 3. PDF search (RAG)
# Uses the same pipeline as the RAG tutorial. The index is built once, lazily,
# from the first .pdf found in docs/.
_collection = None

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    import ollama, chromadb
    from pypdf import PdfReader

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        return None
    reader = PdfReader(pdfs[0])
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + 800]); start += 650
    client = chromadb.Client()
    _collection = client.create_collection("agent_pdf")
    for i in range(0, len(chunks), 32):
        part = chunks[i:i + 32]
        emb = ollama.embed(model="nomic-embed-text", input=part)["embeddings"]
        _collection.add(ids=[str(i + j) for j in range(len(part))], embeddings=emb, documents=part)
    print(f"[search_pdf] indexed {pdfs[0].name}: {len(chunks)} chunks")
    return _collection

def search_pdf(query: str) -> str:
    """Search the PDF in docs/ for passages relevant to the query."""
    import ollama
    col = _get_collection()
    if col is None:
        return "Error: no PDF found in docs/ folder"
    q = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]
    hits = col.query(query_embeddings=[q], n_results=3)
    return "\n\n---\n\n".join(hits["documents"][0])


# ---------------------------------------------------------------- registry
# name -> function. The agent refuses any tool name not in here.
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "read_file": read_file,
    "search_pdf": search_pdf,
}

# What the model sees. Descriptions matter: vague descriptions = wrong tool choice.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression exactly. Use for ANY arithmetic, even simple.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. '847 * 293'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the docs folder by file name.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "e.g. 'notes.txt'"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pdf",
            "description": "Search the PDF document for information. Use when asked about the document's contents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "what to look for"}},
                "required": ["query"],
            },
        },
    },
]
