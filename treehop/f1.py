import jsonlines
from tqdm.auto import tqdm

from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever
import numpy as np

def compute_prf(
    retrieved_sets: list[set[str]],
    gold_sets: list[set[str]]
):
    metrics = []

    for retrieved, gold in zip(retrieved_sets, gold_sets):
        tp = len(retrieved & gold)
        fp = len(retrieved - gold)
        fn = len(gold - retrieved)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        metrics.append((precision, recall, f1))

    return np.mean(metrics, axis=0)


def main():
    EVALUATE_DATASET = "2wiki"

    # load TreeHop model from HuggingFace
    tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")

    # load retriever
    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages=f"embedding_data/{EVALUATE_DATASET}/eval_passages_10.jsonl",
        passage_embeddings=f"embedding_data/{EVALUATE_DATASET}/eval_content_dense.npy",
        # uncomment this if faiss index is initialized, resulting in a faster loading
        faiss_index=f"embedding_data/{EVALUATE_DATASET}/index.faiss",
        tree_hop_model=tree_hop_model,
        projection_size=1024,
        save_or_load_index=True,
        indexing_batch_size=10240,
        index_device="cuda"     # or cpu on Apple Metal
    )

    dataset_file = "eval_data/2wiki_dev_processed_10.jsonl"
    retrieved_sets = []
    gold_sets = []

    with jsonlines.open(dataset_file) as reader:
        for sample in reader:
            question = sample["question"]
            gold_sets.append(set(sf[0] for sf in sample["supporting_facts"]))
            
            retrieve_result = retriever.multihop_search_passages(
                question,
                n_hop=3,
                top_n=20,
                index_batch_size=10240,
                generate_batch_size=1024,
                # redundant_pruning=True,
                # layerwise_top_pruning=True
            )
            last_hop_passages = retrieve_result.passage[-1][0]
            retrieved_titles = {p["title"] for p in last_hop_passages}
            retrieved_sets.append(retrieved_titles)

    # print('\nRETRIEVED SETS', retrieved_sets, '\n')
    # print('\nGOLD SETS', gold_sets, '\n')

    print('\n', len(retrieved_sets), len(gold_sets))

    print()
    for retrieved, gold in zip(retrieved_sets, gold_sets):
        print(retrieved)
        print(gold)
        print()

    precision, recall, f1 = compute_prf(retrieved_sets, gold_sets)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    main()