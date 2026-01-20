import json
import itertools
import argparse

def load_first_n(path, n=10):
    """Load the first n items from a .json or .jsonl file."""
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in itertools.islice(f, n):
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {e}")
        return items

    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[:n]
            elif isinstance(data, dict):
                # Take first n key-value pairs if dict
                return dict(itertools.islice(data.items(), n))
            else:
                raise ValueError("Unsupported JSON structure: expected list or dict.")

def save_json(data, output_path):
    """Save JSON data to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} items to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract the first N items from a JSON or JSONL file.")
    parser.add_argument("input_file", help="Path to input .json or .jsonl file")
    parser.add_argument("output_file", help="Path to save the output .json file")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of items to extract (default=10)")
    args = parser.parse_args()

    first_items = load_first_n(args.input_file, args.num)
    save_json(first_items, args.output_file)

if __name__ == "__main__":
    main()