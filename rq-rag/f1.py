import json
import numpy as np


def compute_prf_metrics(retrieved_titles, gold_titles):
    """
    Standard PRF + EM logic.
    Includes the 'Perfect Empty' case from the RQ-RAG source.
    """
    if not retrieved_titles and not gold_titles:
        return 1.0, 1.0, 1.0, 1.0  # Precision, Recall, F1, EM

    tp = len(retrieved_titles & gold_titles)
    fp = len(retrieved_titles - gold_titles)
    fn = len(gold_titles - retrieved_titles)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    em = 1.0 if (fp == 0 and fn == 0) else 0.0

    return prec, recall, f1, em


def main():
    RESULTS_PATH = "output/dev_5/final_results.json"
    INPUT_DATA_PATH = "data/2wiki/dev_5.json"

    K = 5

    # --- LOAD DATA ---
    with open(RESULTS_PATH, "r") as f:
        results_json = json.load(f)

    with open(INPUT_DATA_PATH, "r") as f:
        input_data = json.load(f)

    # all_results is a list of trees (one per question)
    all_trees = results_json.get("all_results", [])

    all_metrics = []

    print(f"Evaluating {len(all_trees)} samples at K={K}...")

    for i, question_tree in enumerate(all_trees):
        # 1. Get the gold titles for this specific question
        # Supporting facts in your input are [Title, SentenceIndex]
        gold_titles = set(fact[0] for fact in input_data[i]["supporting_facts"])

        # 2. Get the corpus for this question to map indices to titles
        # (RQ-RAG uses the local 'context' provided in the input)
        local_context = input_data[i]["context"]

        # 3. Aggregate all unique titles across the entire tree
        tree_retrieved_titles = []
        seen = set()

        for node in question_tree:
            indices = node.get("retrieved_index", [])
            for idx in indices:
                # Basic safety check for index
                if idx < len(local_context):
                    title = local_context[idx][0]  # Index 0 is the Title
                    if title not in seen:
                        seen.add(title)
                        tree_retrieved_titles.append(title)

        # 4. Apply K-cutoff and convert to set
        final_retrieved_set = set(tree_retrieved_titles[:K])

        # 5. Calculate Metrics
        metrics = compute_prf_metrics(final_retrieved_set, gold_titles)
        all_metrics.append(metrics)

    # --- AGGREGATE ---
    avg_metrics = np.mean(all_metrics, axis=0)

    print(
        f"Precision@{K}: {avg_metrics[0]:.4f}, Recall@{K}: {avg_metrics[1]:.4f}, F1: {avg_metrics[2]:.4f}, EM: {avg_metrics[3]}"
    )


if __name__ == "__main__":
    main()
