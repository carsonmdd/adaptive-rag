import json


def split_2wiki_by_complexity(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    simple_data = []
    complex_data = []

    for item in data:
        # Measure hop count via number of unique supporting paragraphs
        # facts are [title, sent_idx], so we take unique titles
        unique_titles = set(fact for fact in item.get("supporting_facts", []))
        hop_count = len(unique_titles)
        q_type = item.get("type", "")

        # Define 'Simple': 2-hop bridge questions
        if q_type == "bridge" and hop_count <= 2:
            simple_data.append(item)

        # Define 'Complex': 3+ hops OR specific difficult types
        elif hop_count >= 3 or q_type in ["compositional", "comparison", "inference"]:
            complex_data.append(item)

    # Save the split datasets
    with open("2wiki_simple.json", "w") as f:
        json.dump(simple_data, f, indent=4)

    with open("2wiki_complex.json", "w") as f:
        json.dump(complex_data, f, indent=4)

    print(f"Split complete: {len(simple_data)} simple, {len(complex_data)} complex.")


split_2wiki_by_complexity("dev.json")
