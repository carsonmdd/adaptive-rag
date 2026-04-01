import json
import os
import sys


def load_data(file_path):
    """Detects format and loads either .json or .jsonl"""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)


def save_data(data, file_path):
    """Saves data as .json with pretty printing"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def split_2wiki_by_complexity(base_url):
    # Check for either dev.json or dev.jsonl
    json_path = os.path.join(base_url, "dev.json")
    jsonl_path = os.path.join(base_url, "2wiki_dev_processed.jsonl")

    if os.path.exists(jsonl_path):
        target_path = jsonl_path
    elif os.path.exists(json_path):
        target_path = json_path
    else:
        print(
            f"Error: Could not find dev.json or 2wiki_dev_processed.jsonl in {base_url}"
        )
        return

    data = load_data(target_path)
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

        # Criteria: Simple (2-hop Bridge types)
        if hop_count <= 2 and q_type in ["bridge", "inference", "compositional"]:
            simple_data.append(item)

        # Criteria: Complex (3+ hops OR Comparison types)
        elif hop_count >= 3 or q_type in ["comparison", "bridge_comparison"]:
            complex_data.append(item)

    # Save outputs
    save_data(simple_data, os.path.join(base_url, "2wiki_simple.json"))
    save_data(complex_data, os.path.join(base_url, "2wiki_complex.json"))

    print(f"File processed: {os.path.basename(target_path)}")
    print(f"Split complete: {len(simple_data)} simple, {len(complex_data)} complex.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        split_2wiki_by_complexity(target_dir)
    else:
        print("No path provided")
