import json
import os
import torch
from openai import OpenAI
import sys
sys.path.append(os.getcwd())
import retrieval_lm.src.normalize_text as normalize_text

def embed_contexts(input_json_path, output_embeddings_path):
    """
    Reads 2Wiki JSON data, embeds the contexts using OpenAI embeddings,
    and saves them to a torch .pt file.
    """

    client = OpenAI()
    context_embeddings = []

    # Load the 2Wiki data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Loop through each sample
    for sample in data:
        sample_embeddings = []

        # Flatten all context paragraphs for this sample
        for title, sentences in sample["context"]:
            paragraph_text = " ".join(sentences)
            normalized_text = normalize_text.normalize(paragraph_text.lower())

            # Get the embedding from OpenAI
            try:
                emb = client.embeddings.create(
                    input=[normalized_text],
                    model="text-embedding-3-large"
                ).data[0].embedding
            except Exception as e:
                print(f"Warning: failed to embed paragraph '{title}', using dummy embedding. Error: {e}")
                emb = client.embeddings.create(input=["dummy"], model="text-embedding-3-large").data[0].embedding

            sample_embeddings.append(torch.tensor(emb))

        context_embeddings.append(sample_embeddings)

    # Save all embeddings
    torch.save(context_embeddings, output_embeddings_path)
    print(f"Saved context embeddings to {output_embeddings_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to 2Wiki JSON file")
    parser.add_argument("output_embeddings", help="Path to save the embeddings .pt file")
    args = parser.parse_args()

    embed_contexts(args.input_json, args.output_embeddings)
