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
