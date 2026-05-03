from concurrent.futures import ThreadPoolExecutor, as_completed

from passage_retrieval import multihop_search_passage_results
from src.language_models import LanguageModel

_SYSTEM_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer questions concisely and accurately using only the provided passages."
)

_READER_PROMPT = """\
Answer the following question using only the provided passages. \
Give a concise answer — a short phrase or sentence is preferred.

Question: {question}

Passages:
{passages}

Answer:"""


def _collect_passages(
    retrieve_result: multihop_search_passage_results, query_idx: int
) -> str:
    seen = set()
    unique = []
    for hop_passages in retrieve_result.passage:
        if query_idx >= len(hop_passages):
            continue
        for p in hop_passages[query_idx]:
            key = p.get("id", p.get("title", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)

    lines = []
    for i, p in enumerate(unique):
        lines.append(f"[{i + 1}] Title: {p.get('title', '')}\n{p.get('text', '')}")
    return "\n\n".join(lines)


class Reader:
    def __init__(self, llm: LanguageModel):
        self.llm = llm

    def answer(
        self,
        question: str,
        retrieve_result: multihop_search_passage_results,
        query_idx: int = 0,
    ) -> str:
        passages_text = _collect_passages(retrieve_result, query_idx)
        prompt = _READER_PROMPT.format(question=question, passages=passages_text)
        return self.llm.chat(prompt, system_msg=_SYSTEM_PROMPT)

    def answer_batch(
        self,
        questions: list[str],
        retrieve_result: multihop_search_passage_results,
        max_workers: int = 4,
    ) -> list[str]:
        results = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.answer, q, retrieve_result, i): i
                for i, q in enumerate(questions)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results
