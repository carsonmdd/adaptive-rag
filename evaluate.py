"""
End-to-end QA evaluation for RQ-RAG, TreeHop+Reader, and Adaptive RAG.

Metrics
-------
- Token F1: token-overlap F1 between normalized prediction and gold answer
- Latency: wall-clock seconds per question

Usage
-----
python evaluate.py --system [rqrag|treehop|adaptive] \
                   --data rq-rag/data/2wiki/2wiki_simple_500.json \
                   --output results/treehop_simple.json

python evaluate.py --system adaptive \
                   --data rq-rag/data/2wiki/2wiki_complex_500.json \
                   --output results/adaptive_complex.json \
                   --limit 50   # optional: first N questions
"""

import argparse
import json
import os
import string
import sys
import time
from collections import Counter, defaultdict

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

_root = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    articles = {"a", "an", "the"}
    return " ".join(t for t in tokens if t not in articles)


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Pipeline loaders
# ---------------------------------------------------------------------------

def _load_treehop_components(data_file: str):
    _treehop = os.path.join(_root, "treehop")
    sys.path.insert(0, _treehop)
    from tree_hop import TreeHopModel
    from passage_retrieval import MultiHopRetriever
    from reader import Reader
    from src.language_models import OpenAIModel

    sys.path.remove(_treehop)
    for mod in ("utils", "metrics", "f1", "evaluation"):
        sys.modules.pop(mod, None)

    stem = os.path.splitext(os.path.basename(data_file))[0]
    index_dir = os.path.join(_root, "treehop", "embedding_data", stem)
    passages = os.path.join(index_dir, "eval_passages.jsonl")
    faiss_index = os.path.join(index_dir, "index.faiss")
    emb_path = os.path.join(index_dir, "eval_content_dense.npy")

    retriever_kwargs = dict(faiss_index=faiss_index) if os.path.exists(faiss_index) \
        else dict(passage_embeddings=emb_path)

    model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")
    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages=passages,
        tree_hop_model=model,
        projection_size=1024,
        save_or_load_index=True,
        index_device="cuda",
        **retriever_kwargs,
    )
    reader = Reader(OpenAIModel())
    return retriever, reader


def _load_rqrag(data_file: str):
    _rqrag = os.path.join(_root, "rq-rag")
    _rqrag_lm = os.path.join(_root, "rq-rag", "retrieval_lm")
    sys.path.insert(0, _rqrag)
    sys.path.insert(0, _rqrag_lm)
    from pipeline import RQRAGPipeline
    return RQRAGPipeline("zorowin123/rq_rag_llama2_7B", input_file=data_file)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(system: str, data_file: str, output_file: str, limit: int = None):
    with open(data_file) as f:
        items = json.load(f)
    if limit:
        items = items[:limit]

    # Build the requested pipeline
    retriever = reader = rqrag = adaptive_pipeline = None

    if system == "treehop":
        retriever, reader = _load_treehop_components(data_file)

    elif system == "rqrag":
        rqrag = _load_rqrag(data_file)

    elif system == "adaptive":
        retriever, reader = _load_treehop_components(data_file)

        _rqrag_mod = os.path.join(_root, "rq-rag")
        _rqrag_lm = os.path.join(_root, "rq-rag", "retrieval_lm")
        sys.path.insert(0, _rqrag_mod)
        sys.path.insert(0, _rqrag_lm)
        sys.path.insert(0, _root)

        from pipeline import AdaptiveRAGPipeline, RQRAGPipeline
        rqrag_pipe = RQRAGPipeline("zorowin123/rq_rag_llama2_7B", input_file=data_file)
        adaptive_pipeline = AdaptiveRAGPipeline(
            router_path="rag_router/2wiki-type-classifier-final",
            treehop_retriever=retriever,
            reader=reader,
            rqrag_pipeline=rqrag_pipe,
        )
    else:
        raise ValueError(f"Unknown system: {system!r}")

    records = []
    type_buckets = defaultdict(lambda: {"f1": [], "latency": []})
    overall_f1, overall_latency = [], []

    for i, item in enumerate(items):
        t0 = time.perf_counter()

        if system == "treehop":
            retrieve_result = retriever.multihop_search_passages(
                item["question"], n_hop=2, top_n=5
            )
            pred = reader.answer(item["question"], retrieve_result)
            route = "treehop"
        elif system == "rqrag":
            pred = rqrag.answer(
                item["question"], context=item.get("context"), question_idx=i
            )
            route = "rqrag"
        else:
            result = adaptive_pipeline.answer(
                item["question"], context=item.get("context"), question_idx=i
            )
            pred = result["answer"]
            route = result["path"]

        latency = time.perf_counter() - t0
        f1 = compute_f1(pred, item["answer"])
        qtype = item["type"]

        record = {
            "idx": i,
            "id": item.get("_id", ""),
            "type": qtype,
            "question": item["question"],
            "gold": item["answer"],
            "pred": pred,
            "route": route,
            "f1": f1,
            "latency": latency,
        }
        records.append(record)
        type_buckets[qtype]["f1"].append(f1)
        type_buckets[qtype]["latency"].append(latency)
        overall_f1.append(f1)
        overall_latency.append(latency)

        print(
            f"[{i:4d}] {qtype:20s} → {route:7s}  "
            f"F1={f1:.2f}  ({latency:.1f}s)"
        )
        print(f"       pred: {pred[:80]}")
        print(f"       gold: {item['answer']}")

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    by_type = {
        qtype: {
            "f1": mean(v["f1"]),
            "avg_latency": mean(v["latency"]),
            "n": len(v["f1"]),
        }
        for qtype, v in type_buckets.items()
    }

    summary = {
        "system": system,
        "data_file": data_file,
        "n": len(records),
        "overall_f1": mean(overall_f1),
        "avg_latency": mean(overall_latency),
        "by_type": by_type,
        "records": records,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"System: {system}   N={len(records)}")
    print(f"Overall F1={summary['overall_f1']:.3f}  Avg latency={summary['avg_latency']:.1f}s/q")
    for qtype, m in sorted(by_type.items()):
        print(f"  {qtype:22s}  F1={m['f1']:.3f}  n={m['n']}  lat={m['avg_latency']:.1f}s")
    print(f"\nSaved → {output_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["rqrag", "treehop", "adaptive"], required=True)
    parser.add_argument("--data", required=True, help="Path to JSON eval file")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N questions only")
    args = parser.parse_args()

    evaluate(args.system, args.data, args.output, args.limit)
