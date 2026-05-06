import json

from pipeline import AdaptiveRAGPipeline, RQRAGPipeline
from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever
from src.language_models import OpenAIModel
from reader import Reader

tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")
retriever = MultiHopRetriever(
    "BAAI/bge-m3",
    passages="treehop/embedding_data/2wiki/eval_passages.jsonl",
    faiss_index="treehop/embedding_data/2wiki/index.faiss",
    tree_hop_model=tree_hop_model,
    projection_size=1024,
    save_or_load_index=True,
    index_device="cuda",
)

EVAL_FILE = "rq-rag/data/2wiki/dev_5.json"

pipeline = AdaptiveRAGPipeline(
    router_path="rag_router/2wiki-type-classifier-final",
    treehop_retriever=retriever,
    reader=Reader(OpenAIModel()),
    rqrag_pipeline=RQRAGPipeline("zorowin123/rq_rag_llama2_7B", input_file=EVAL_FILE),
)

with open(EVAL_FILE) as f:
    questions = json.load(f)

for i, q in enumerate(questions):
    result = pipeline.answer(q["question"], context=q["context"], question_idx=i)
    correct = result["answer"].strip().lower() == q["answer"].strip().lower()
    print(f"[{result['type']:20s} → {result['path']:7s}] {'✓' if correct else '✗'}  Q: {q['question']}")
    print(f"   predicted: {result['answer']}")
    print(f"   expected:  {q['answer']}")
    print()
