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

pipeline = AdaptiveRAGPipeline(
    router_path="rag_router/2wiki-type-classifier-final",
    treehop_retriever=retriever,
    reader=Reader(OpenAIModel()),
    rqrag_pipeline=RQRAGPipeline("zorowin123/rq_rag_llama2_7B"),
)

print(
    pipeline.answer("Which film came out first, Blind Shaft or The Mask Of Fu Manchu?")
)
# --> {"type": "comparison", "path": "treehop", "answer": "The Mask Of Fu Manchu"}

print(pipeline.answer("Who is the mother of the director of Polish-Russian War?"))
# --> {"type": "compositional", "path": "rqrag", "answer": "Małgorzata Braunek"}
