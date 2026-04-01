import json
import os
import sys


def split_2wiki_by_complexity(base_url):
    with open(os.path.join(base_url, "dev.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    simple_data = []
    complex_data = []

    for item in data:
        # Measure hop count via number of unique supporting paragraphs
        # facts are [title, sent_idx], so we take unique titles
        unique_titles = set(
            title for title, sent_idx in item.get("supporting_facts", [])
        )
        hop_count = len(unique_titles)
        q_type = item.get("type", "")

        # Simple Dataset:
        # 2-hop questions where the logic is a straightforward bridge.
        # Paper states: 'Inference' and 'Compositional' are bridge subtypes.
        if hop_count <= 2 and q_type in ["inference", "compositional"]:
            simple_data.append(item)

        # Complex Dataset:
        # 1. Any question involving 3 or more unique documents (High-Hop).
        # 2. 'Comparison' or 'Bridge-Comparison' (Requires parallel entity processing).
        elif hop_count >= 3 or q_type in ["comparison", "bridge_comparison"]:
            complex_data.append(item)

    # Save the split datasets
    with open(os.path.join(base_url, "2wiki_simple.json"), "w") as f:
        json.dump(simple_data, f, indent=4)

    with open(os.path.join(base_url, "2wiki_complex.json"), "w") as f:
        json.dump(complex_data, f, indent=4)

    print(f"Split complete: {len(simple_data)} simple, {len(complex_data)} complex.")


if __name__ == "__main__":
    # Check if a directory was provided as a parameter
    if len(sys.argv) > 1:
        target_dir = sys.argv
        split_2wiki_by_complexity(target_dir)
    else:
        print("No path provided")
