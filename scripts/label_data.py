import jsonlines
import random
from treehop.tree_hop import TreeHopModel
from treehop.passage_retrieval import MultiHopRetriever


def run_treehop(query, gold_set):
    EVALUATE_DATASET = "2wiki"

    # load TreeHop model from HuggingFace
    tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")

    # load retriever
    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages=f"embedding_data/{EVALUATE_DATASET}/eval_passages.jsonl",
        passage_embeddings=f"embedding_data/{EVALUATE_DATASET}/eval_content_dense.npy",
        # uncomment this if faiss index is initialized, resulting in a faster loading
        # faiss_index=f"embedding_data/{EVALUATE_DATASET}/index.faiss",
        tree_hop_model=tree_hop_model,
        projection_size=1024,
        save_or_load_index=True,
        indexing_batch_size=10240,
        index_device="cuda",  # or cpu on Apple Metal
    )

    retrieve_result = retriever.multihop_search_passages(
        query,
        n_hop=2,
        top_n=5,
    )

    # Extract and flatten candidates from all hops for the first (and only) query
    all_candidates = []
    for hop_passages in retrieve_result.passage:
        all_candidates.extend(hop_passages)

    # Sort by score
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    # De-duplicate and filter to Top-K
    K = 2
    retrieved_titles = []
    seen = set()
    for cand in all_candidates:
        if cand["title"] not in seen:
            seen.add(cand["title"])
            retrieved_titles.append(cand["title"])
        if len(retrieved_titles) >= K:
            break

    # Calculate Recall
    retrieved_set = set(retrieved_titles)
    tp = len(retrieved_set & gold_set)
    fn = len(gold_set - retrieved_set)
    recall = tp / (tp + fn) if tp + fn else 0

    return recall


def run_rq_rag(query, gold_set):
    # RQ-RAG logic here
    # Return recall_score
    pass


def create_router_dataset(source_file, output_file, samples_needed=2000):
    dataset = load_data(source_file)
    random.seed(42)
    random.shuffle(dataset)

    labeled_data = []

    for item in dataset:
        if len(labeled_data) >= samples_needed:
            break

        query = item["question"]
        gold = set(sf for sf in item["supporting_facts"])

        # 1. Test Fast Path (TreeHop)
        th_recall = run_treehop(query, gold)
        if th_recall >= 0.8:
            labeled_data.append({"text": query, "label": 0})  # Label 0 = TreeHop
            continue

        # 2. Test Precision Path (RQ-RAG) only if TreeHop failed
        rq_recall = run_rq_rag(query, gold)
        if rq_recall >= 0.8:
            labeled_data.append({"text": query, "label": 1})  # Label 1 = RQ-RAG

    save_jsonl(labeled_data, output_file)
