import jsonlines
from tqdm.auto import tqdm

from treehop.tree_hop import TreeHopModel
from treehop.passage_retrieval import MultiHopRetriever


EVALUATE_DATASET = "2wiki"

# load TreeHop model from HuggingFace
tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")

# load retriever
retriever = MultiHopRetriever(
    "BAAI/bge-m3",
    passages=f"embedding_data/{EVALUATE_DATASET}/eval_passages_10.jsonl",
    passage_embeddings=f"embedding_data/{EVALUATE_DATASET}/eval_content_dense.npy",
    # uncomment this if faiss index is initialized, resulting in a faster loading
    # faiss_index=f"embedding_data/{EVALUATE_DATASET}/index.faiss",
    tree_hop_model=tree_hop_model,
    projection_size=1024,
    save_or_load_index=True,
    indexing_batch_size=10240,
    index_device="cuda"     # or cpu on Apple Metal
)

def get_gold_passages(sample):
    """Return set of supporting passage titles for one question"""
    return set([sf[0] for sf in sample["supporting_facts"]])


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


# dataset_file = "treehop/eval_data/2wiki_dev_processed_10.jsonl"
# retrieved_sets = []
# gold_sets = []

# with jsonlines.open(dataset_file) as reader:
#     for sample in reader:
#         question = sample["question"]
#         gold_sets.append(set(sf[0] for sf in sample["supporting_facts"]))
        
#         retrieve_result = retriever.multihop_search_passages(
#             question,
#             n_hop=2,
#             top_n=5,
#             return_tree=True
#         )
#         retrieval_tree = retrieve_result.tree_hop_graph[0]
#         retrieved_sets.append(set(node_data['title'] for node_id, node_data in retrieval_tree.nodes(data=True)))

# precision, recall, f1 = compute_prf(retrieved_sets, gold_sets)

# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
