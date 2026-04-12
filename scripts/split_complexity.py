import json
import os
import sys
import jsonlines
import random


def load_data(file_path):
    """Detects format and loads either .json or .jsonl"""
    if file_path.endswith(".jsonl"):
        with jsonlines.open(file_path) as reader:
            return list(reader)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def save_data(data, file_path):
    """Saves data as .jsonl or .json based on extension"""
    if file_path.endswith(".jsonl"):
        with jsonlines.open(file_path, mode="w") as writer:
            writer.write_all(data)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def split_2wiki_by_complexity(base_url, limit=None):
    random.seed(42)

    # Check for either dev.json or the processed jsonl
    json_path = os.path.join(base_url, "dev.json")
    jsonl_path = os.path.join(base_url, "2wiki_dev_processed.jsonl")

    if os.path.exists(jsonl_path):
        target_path = jsonl_path
        ext = ".jsonl"  # Save output in the same format
    elif os.path.exists(json_path):
        target_path = json_path
        ext = ".json"
    else:
        print(
            f"Error: Could not find dev.json or 2wiki_dev_processed.jsonl in {base_url}"
        )
        return

    data = load_data(target_path)
    random.shuffle(data)

    simple_data = []
    complex_data = []

    for item in data:
        # Get unique titles to count true hops
        unique_titles = set(
            title for title, sent_idx in item.get("supporting_facts", [])
        )
        hop_count = len(unique_titles)
        q_type = item.get("type", "").lower()

        # Criteria: Simple (2-hop comparison or inference)
        if hop_count <= 2 and q_type in ["comparison", "inference"]:
            simple_data.append(item)

        # Criteria: Complex (3+ hops OR compositional or bridge_comparison)
        elif hop_count >= 3 or q_type in ["compositional", "bridge_comparison"]:
            complex_data.append(item)

    if limit:
        simple_data = simple_data[:limit]
        complex_data = complex_data[:limit]

    # Save outputs using the detected extension
    limit_suffix = f"_{limit}" if limit else ""
    save_data(simple_data, os.path.join(base_url, f"2wiki_simple{limit_suffix}{ext}"))
    save_data(complex_data, os.path.join(base_url, f"2wiki_complex{limit_suffix}{ext}"))

    print(f"--- Processing Complete ---")
    print(f"Source: {os.path.basename(target_path)}")
    print(f"Sampled with Seed 42 and Limit: {limit}")
    print(f"Format: {ext}")
    print(f"Simple: {len(simple_data)}")
    print(f"Complex: {len(complex_data)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        n_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        split_2wiki_by_complexity(target_dir, n_limit)
    else:
        print(
            "Usage: python scripts/split_complexity.py /home/path/to/data/ [optional_limit]"
        )
