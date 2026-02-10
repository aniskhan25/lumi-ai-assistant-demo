#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Doc:
    path: str
    name: str
    text: str
    tfidf: Dict[str, float]
    norm: float


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_tfidf(docs_text: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    term_doc_freq = Counter()
    doc_terms = []
    for text in docs_text:
        tokens = tokenize(text)
        counts = Counter(tokens)
        doc_terms.append(counts)
        for term in counts:
            term_doc_freq[term] += 1

    n_docs = len(docs_text)
    idf = {}
    for term, df in term_doc_freq.items():
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    tfidf_docs = []
    for counts in doc_terms:
        doc_vec = {}
        for term, tf in counts.items():
            doc_vec[term] = tf * idf.get(term, 0.0)
        tfidf_docs.append(doc_vec)

    return tfidf_docs, idf


def vec_norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine_similarity(a: Dict[str, float], a_norm: float, b: Dict[str, float], b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    dot = 0.0
    for term, v in a.items():
        bv = b.get(term)
        if bv:
            dot += v * bv
    return dot / (a_norm * b_norm)


def load_docs(docs_dir: str) -> List[Doc]:
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    paths = []
    for name in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, name)
        if os.path.isfile(path) and name.lower().endswith((".md", ".txt")):
            paths.append(path)

    if not paths:
        raise FileNotFoundError(f"No .md or .txt files found in {docs_dir}")

    texts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    tfidf_docs, _idf = build_tfidf(texts)
    docs = []
    for path, text, vec in zip(paths, texts, tfidf_docs):
        docs.append(
            Doc(
                path=path,
                name=os.path.basename(path),
                text=text,
                tfidf=vec,
                norm=vec_norm(vec),
            )
        )
    return docs


def retrieve(docs: List[Doc], query: str, k: int) -> List[Doc]:
    q_counts = Counter(tokenize(query))
    # Build query vector using doc IDF derived from docs
    term_doc_freq = Counter()
    for doc in docs:
        for term in doc.tfidf.keys():
            term_doc_freq[term] += 1
    n_docs = len(docs)
    q_vec = {}
    for term, tf in q_counts.items():
        df = term_doc_freq.get(term, 0)
        idf = math.log((n_docs + 1) / (df + 1)) + 1.0
        q_vec[term] = tf * idf
    q_norm = vec_norm(q_vec)

    scored = []
    for doc in docs:
        score = cosine_similarity(q_vec, q_norm, doc.tfidf, doc.norm)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k] if score > 0.0]


def http_request_json(method: str, url: str, payload: dict = None, timeout: float = 30.0) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def get_model_id(base_url: str) -> str:
    models = http_request_json("GET", f"{base_url}/models")
    data = models.get("data", [])
    if not data:
        raise RuntimeError("No models returned from /v1/models")
    return data[0].get("id")


def slurm_template(account: str, partition: str, gpus: int, hours: int) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=vllm-demo
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output=demo-%j.out
#SBATCH --error=demo-%j.err

# Load modules / set env as needed
# module load ...

srun python demo_agent.py
""".strip()


def detect_tool_output(question: str) -> str:
    q = question.lower()
    if any(tok in q for tok in ["sbatch", "slurm", "job script", "batch script", "gpu job"]):
        account = os.environ.get("ACCOUNT", "YOUR_ACCOUNT")
        partition = os.environ.get("PARTITION", "standard-g")
        gpus = int(os.environ.get("GPUS", "1"))
        hours = int(os.environ.get("HOURS", "1"))
        return slurm_template(account, partition, gpus, hours)
    return ""


def build_prompt(question: str, retrieved: List[Doc], tool_output: str) -> List[dict]:
    context_blocks = []
    for doc in retrieved:
        context_blocks.append(f"[source: {doc.name}]\n{doc.text.strip()}")
    context = "\n\n".join(context_blocks)

    system = (
        "You are a LUMI demo assistant. Use only the provided context to answer. "
        "If the context is insufficient, say so and suggest what to check next. "
        "Cite sources by filename in the format (source: file.md). "
        "Keep the answer concise and structured as: short answer, steps, citations."
    )

    user = """Context:
{context}

Tool output (if any):
{tool}

Question:
{question}
""".format(
        context=context if context else "(no relevant context retrieved)",
        tool=tool_output if tool_output else "(none)",
        question=question,
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chat(base_url: str, model: str, messages: List[dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = http_request_json("POST", f"{base_url}/chat/completions", payload)
    choices = resp.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned from chat completion")
    return choices[0]["message"]["content"].strip()


def read_questions_from_file(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(line)
    return questions


def run_single_question(question: str, docs: List[Doc], base_url: str, model: str, k: int):
    retrieved = retrieve(docs, question, k)
    tool_output = detect_tool_output(question)

    print("\n=== Question ===")
    print(question)
    print("\nRetrieved docs:", ", ".join(d.name for d in retrieved) if retrieved else "(none)")

    messages = build_prompt(question, retrieved, tool_output)
    answer = chat(base_url, model, messages)

    if tool_output:
        print("\n--- Tool Output (Slurm Template) ---")
        print(tool_output)

    print("\n--- Answer ---")
    print(answer)


def main() -> int:
    parser = argparse.ArgumentParser(description="LUMI vLLM demo agent")
    parser.add_argument("--docs", default="./lumi_docs", help="Path to docs directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI base URL")
    parser.add_argument("--model", default=os.environ.get("MODEL"), help="Model ID override")
    parser.add_argument("--top-k", type=int, default=3, help="Number of docs to retrieve")
    parser.add_argument("--question", help="Single question to answer")
    parser.add_argument("--question-file", help="File with one question per line")
    args = parser.parse_args()

    try:
        docs = load_docs(args.docs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2

    try:
        model = args.model or get_model_id(args.base_url)
    except Exception as e:
        print(f"Error: failed to get model id from {args.base_url}: {e}")
        return 3

    print(f"Using model: {model}")

    if args.question_file:
        questions = read_questions_from_file(args.question_file)
        if not questions:
            print("Error: no questions found in question file")
            return 4
        for q in questions:
            run_single_question(q, docs, args.base_url, model, args.top_k)
        return 0

    if args.question:
        run_single_question(args.question, docs, args.base_url, model, args.top_k)
        return 0

    print("Enter questions (type 'exit' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        run_single_question(q, docs, args.base_url, model, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
