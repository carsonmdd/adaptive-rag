import json
import os
from dotenv import load_dotenv
import faiss
from openai import OpenAI
import torch
import numpy as np
import sys

sys.path.append("../..")
import retrieval_lm.src.normalize_text as normalize_text

load_dotenv()


class OpenAIEmbedSearch:
    def __init__(
        self,
        ndocs,
        task,
        args,
        use_calculated_embeds=True,
        is_train=False,
        global_corpus=False,
    ):
        self.ndocs = ndocs
        self.task = task
        self.args = args
        self.client = OpenAI()
        self.global_corpus = global_corpus
        self.corpus = None

        # set the question embedding save path and retrieval results save path
        if use_calculated_embeds:

            if is_train:

                embeddings_path = os.path.join(
                    os.path.dirname(self.args.input_file), "train_context_embeddings.pt"
                )
                self.all_embeddings = torch.load(embeddings_path)

            else:

                stem = os.path.splitext(os.path.basename(self.args.input_file))[0]
                embeddings_path = os.path.join(
                    os.path.dirname(self.args.input_file), f"{stem}_embeddings.pt"
                )
                self.all_embeddings = torch.load(embeddings_path)

                if self.global_corpus:
                    self._build_global_corpus()

    def _build_global_corpus(self):
        """Flatten the per-question [n_questions, n_ctx, dim] embeddings into one
        deduplicated [n_passages, dim] corpus shared by every query.

        Embedding row (q, c) corresponds to question q's context c in the input
        file, in order — the same order generate_embeddings_sag.py embedded them.
        Padding rows (questions with fewer than n_ctx contexts) are skipped by
        walking each question's actual context list.
        """
        with open(self.args.input_file) as f:
            items = json.load(f)

        seen = {}
        corpus, rows = [], []
        for q_idx, item in enumerate(items):
            for c_idx, (title, sentences) in enumerate(item["context"]):
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                key = (title, text)
                if key in seen:
                    continue
                seen[key] = len(corpus)
                corpus.append({"title": title, "paragraph_text": text})
                rows.append(self.all_embeddings[q_idx, c_idx])

        self.corpus = corpus
        self.all_embeddings = torch.stack(rows)

    def __call__(self, query: str, corpus: list, index: int = None):

        if self.global_corpus:
            # search the whole deduplicated corpus, not the question's own contexts
            corpus = self.corpus
            index = None

        # get the normalized text and embedding

        normalized_query = query.lower()
        normalized_query = normalize_text.normalize(normalized_query)

        try:
            normalized_query_emb = (
                self.client.embeddings.create(
                    input=[normalized_query], model="text-embedding-3-large"
                )
                .data[0]
                .embedding
            )

        except Exception as E:
            print(f"bad request for openai, use dummy input : {E}")
            normalized_query_emb = (
                self.client.embeddings.create(
                    input=["dummy"], model="text-embedding-3-large"
                )
                .data[0]
                .embedding
            )

        normalized_query_emb = torch.tensor(normalized_query_emb)

        top_indices = self.find_most_similar_context(normalized_query_emb, index)

        evidences = []
        for top_index in top_indices:
            doc = corpus[top_index]

            # 2Wiki Format: ["Title", ["Sentence 1", "Sentence 2", ...]]
            if isinstance(doc, list) and len(doc) >= 2:
                title = doc[0]
                sentences = doc[1]
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                evidences.append({"title": title, "text": text})
            elif isinstance(doc, dict):
                if "title" in doc:
                    evidences.append(
                        {
                            "title": doc["title"],
                            "text": doc["paragraph_text"],
                        }
                    )
                elif "text" in doc:
                    evidences.append(
                        {
                            "title": "Retrieved Documents for Reference",
                            "text": doc["text"].split("*****")[-1],
                        }
                    )

        if len(evidences) == 0:
            # do not return anything from search engine, add dummy
            print("\n", "HERE", "\n")
            evidences.append(
                {"title": "dummy", "text": "the search engine did not return anything"}
            )

        return evidences, top_indices

    def cosine_similarity(self, a, b):
        """calc the similarity of a and b"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def find_most_similar_context(self, query_embedding, cur_index):
        """find the most relevant context

        parameter:
        query_embedding -- query embedding (1D numpy array)
        context_embeddings -- context list (2D numpy array)

        return:
        the most similar index and embedding
        """

        if cur_index is not None:
            cur_context_embeddings = self.all_embeddings[cur_index]
        else:
            cur_context_embeddings = self.all_embeddings

        emb = cur_context_embeddings.float()
        query = query_embedding.float()
        similarities = (emb @ query) / (emb.norm(dim=1) * query.norm() + 1e-10)

        k = min(self.ndocs, similarities.shape[0])
        return torch.topk(similarities, k).indices.tolist()
