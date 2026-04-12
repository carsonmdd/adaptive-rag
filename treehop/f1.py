import jsonlines

from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever
import numpy as np


def compute_prf(retrieved_sets: list[set[str]], gold_sets: list[set[str]]):
    metrics = []

    for retrieved, gold in zip(retrieved_sets, gold_sets):
        tp = len(retrieved & gold)
        fp = len(retrieved - gold)
        fn = len(gold - retrieved)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        metrics.append((precision, recall, f1))

    return np.mean(metrics, axis=0)


def main():
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

    dataset_file = "eval_data/2wiki_complex_500.jsonl"
    gold_sets = []
    questions = []
    with jsonlines.open(dataset_file) as reader:
        for sample in reader:
            questions.append(sample["question"])
            gold_sets.append(set(sf[0] for sf in sample["supporting_facts"]))

    retrieve_result = retriever.multihop_search_passages(
        questions,
        n_hop=2,
        top_n=5,
        index_batch_size=2048,
        generate_batch_size=1024,
        redundant_pruning=True,
        layerwise_top_pruning=True,
    )

    # retrieve_result.passage is structured as: [hop_0, hop_1, ... hop_n]
    # Each hop is a list of lists: [query_0_passages, query_1_passages, ...]

    K = 2
    num_queries = len(retrieve_result.passage[0])
    retrieved_sets = []

    for i in range(num_queries):
        all_candidates = []

        # 1. Collect every single passage from every hop with its score
        for hop_passages in retrieve_result.passage:
            all_candidates.extend(hop_passages[i])

        # 2. Sort all candidates by score (highest first)
        # Note: If your scores are 'distances' (lower is better), remove reverse=True
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 3. De-duplicate by title while maintaining the sorted order
        query_titles_list = []
        seen = set()
        for cand in all_candidates:
            if cand["title"] not in seen:
                seen.add(cand["title"])
                query_titles_list.append(cand["title"])

            if len(query_titles_list) >= K:
                break

        retrieved_sets.append(set(query_titles_list))

    print()
    for retrieved, gold in zip(retrieved_sets, gold_sets):
        print(retrieved)
        print(gold)
        print()

    precision, recall, f1 = compute_prf(retrieved_sets, gold_sets)

    print(f"Precision@{K}: {precision:.4f}, Recall@{K}: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
