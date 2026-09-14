"""Source-linked semantic retrieval over a small, synthetic support knowledge base.

New AI-assisted portfolio example, September 2026. Endee itself is upstream code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
MODEL = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    category: str
    text: str


def load_documents(path: Path) -> list[Document]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("Corpus must be a non-empty list")
    docs, ids = [], set()
    for item in raw:
        doc = Document(**item)
        if any(not isinstance(v, str) or not v.strip() for v in asdict(doc).values()):
            raise ValueError("Every document field must be a non-empty string")
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", doc.id) or doc.id in ids:
            raise ValueError("Document IDs must be unique and URL-safe")
        ids.add(doc.id)
        docs.append(doc)
    return docs


def chunks(doc: Document, size: int = 600, overlap: int = 80) -> list[dict]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    result, start = [], 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        text = doc.text[start:end]
        digest = hashlib.sha256(f"{doc.id}:{start}:{text}".encode()).hexdigest()[:24]
        result.append({"id": digest, "document_id": doc.id, "title": doc.title,
                       "category": doc.category, "text": text, "start": start, "end": end,
                       "citation": f"corpus.json#{doc.id}:{start}-{end}"})
        if end == len(doc.text):
            break
        start = end - overlap
    return result


def validate_vector(vector, dimension=DIMENSION) -> list[float]:
    values = [float(x) for x in vector]
    if len(values) != dimension or not all(math.isfinite(x) for x in values):
        raise ValueError("Embedding has incorrect dimension or non-finite values")
    if not any(x != 0 for x in values):
        raise ValueError("Zero embeddings are not valid cosine-search inputs")
    return values


def retrieval_metrics(ranked_ids: list[list[str]], relevant_ids: list[set[str]]) -> dict:
    if not ranked_ids or len(ranked_ids) != len(relevant_ids) or any(not r for r in relevant_ids):
        raise ValueError("Evaluation requires paired queries with non-empty relevance labels")
    top1, recall, reciprocal = [], [], []
    for ranked, relevant in zip(ranked_ids, relevant_ids):
        ranked = list(dict.fromkeys(ranked))
        top1.append(float(bool(ranked) and ranked[0] in relevant))
        recall.append(len(set(ranked) & relevant) / len(relevant))
        reciprocal.append(next((1 / i for i, doc_id in enumerate(ranked, 1) if doc_id in relevant), 0))
    mean = lambda xs: sum(xs) / len(xs)
    return {"queries": len(ranked_ids), "hit_at_1": mean(top1),
            "recall_at_k": mean(recall), "mrr_at_k": mean(reciprocal)}


class EvidenceSearch:
    def __init__(self, index_name: str, base_url: str):
        if not re.fullmatch(r"evidence_[a-zA-Z0-9_]{1,39}", index_name):
            raise ValueError("Use an evidence_ index name, maximum 48 characters")
        from endee import Endee
        from fastembed import TextEmbedding
        self.client = Endee(os.environ.get("NDD_AUTH_TOKEN") or None)
        self.client.set_base_url(base_url.rstrip("/"))
        self.index_name = index_name
        self.model = TextEmbedding(model_name=MODEL, threads=2)

    def ingest(self, documents: list[Document]) -> dict:
        from endee import Precision
        parts = [part for doc in documents for part in chunks(doc)]
        if not parts:
            raise ValueError("No chunks to ingest")
        vectors = list(self.model.embed([p["title"] + ": " + p["text"] for p in parts]))
        if len(vectors) != len(parts):
            raise RuntimeError("Embedding count mismatch")
        records = [{"id": p["id"], "vector": validate_vector(v), "meta": p,
                    "filter": {"category": p["category"]}} for p, v in zip(parts, vectors)]
        # Refuse to replace existing indexes: use a new name for a new corpus/model.
        self.client.create_index(name=self.index_name, dimension=DIMENSION,
                                 space_type="cosine", precision=Precision.FLOAT32)
        index = self.client.get_index(self.index_name)
        for offset in range(0, len(records), 64):
            index.upsert(records[offset:offset + 64])
        return {"documents": len(documents), "chunks": len(records), "model": MODEL,
                "dimension": DIMENSION, "index": self.index_name}

    def search(self, query: str, top_k: int = 3, category: str | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must contain 1–2000 characters")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        vector = validate_vector(next(self.model.query_embed(query)))
        kwargs = {"vector": vector, "top_k": top_k, "ef": 128, "include_vectors": False}
        if category:
            kwargs["filter"] = [{"category": {"$eq": category}}]
        results = self.client.get_index(self.index_name).query(**kwargs)
        citations = []
        for item in results:
            meta = item.get("meta")
            if not isinstance(meta, dict) or not all(k in meta for k in ["citation", "text", "document_id", "category"]):
                raise RuntimeError("Result is missing source metadata; refusing ungrounded output")
            if category and meta["category"] != category:
                raise RuntimeError("Database returned a result outside the requested category")
            citations.append({**meta, "similarity": item["similarity"]})
        return citations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "search", "evaluate"])
    parser.add_argument("--index", default="evidence_support_demo")
    parser.add_argument("--url", default="http://127.0.0.1:8080/api/v1")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.json")
    parser.add_argument("--query", default="")
    parser.add_argument("--category")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = EvidenceSearch(args.index, args.url)
    if args.command == "ingest":
        result = engine.ingest(load_documents(args.corpus))
    elif args.command == "search":
        result = engine.search(args.query, args.top_k, args.category)
    else:
        cases = json.loads((ROOT / "queries.json").read_text())
        ranked, relevant, observations = [], [], []
        for case in cases:
            start = time.perf_counter()
            hits = engine.search(case["query"], args.top_k, case.get("category"))
            elapsed = time.perf_counter() - start
            ranked.append([h["document_id"] for h in hits]); relevant.append(set(case["relevant"]))
            observations.append({**case, "retrieved": ranked[-1], "seconds": elapsed,
                                 "citations": [h["citation"] for h in hits]})
        result = {"model": MODEL, "index": args.index, "top_k": args.top_k,
                  "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                  "metrics": retrieval_metrics(ranked, relevant), "observations": observations,
                  "scope": "Small synthetic demonstration; not a production accuracy benchmark or ATS score."}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
