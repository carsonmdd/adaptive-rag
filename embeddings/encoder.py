import torch
import utils.normalize_text

@torch.no_grad
def embed_queries(self, queries):
    embeddings, batch_query = [], []
    for k, q in enumerate(queries):
        q.lower()
        q = utils.normalize_text.normalize(q)
        batch_query.append(q)

        if len(batch_query) == self.per_gpu_batch_size or k == len(queries) - 1:

            encoded_batch = self.tokenizer.batch_encode_plus(
                batch_query,
                return_tensors="pt",
                max_length=self.query_maxlength,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
            output = self.model(**encoded_batch)
            embeddings.append(output.to(self.index_device))

            batch_query.clear()
            # getattr(torch, DEVICE).empty_cache()

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings