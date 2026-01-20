from embeddings.encoder import embed_query
# from treehop.search import TreeHopSearch
# from routing.confidence import compute_confidence
# from routing.triggers import needs_refinement
# from rqrag.refinement import QueryRefiner
# from generation.answer import AnswerGenerator
from utils.types import *
import sys


def run_pipeline(
    query_text: str,
    *,
    max_hops: int = 3,
    confidence_threshold: float = 0.5,
):
    """
    Entry point for the hybrid RAG system.
    """

    # -------------------------------------------------
    # Step 1: Embed query
    # -------------------------------------------------
    # query_embedding = embed_query(query_text)

    query_embedding = None  # placeholder

    # -------------------------------------------------
    # Step 2: TreeHop retrieval
    # -------------------------------------------------
    # treehop = TreeHopSearch(max_hops=max_hops)
    # treehop_result = treehop.run(query_embedding)

    treehop_result = TreeHopResult(passages=[])

    # -------------------------------------------------
    # Step 3: Compute confidence
    # -------------------------------------------------
    # treehop_result.confidence = compute_confidence(treehop_result)

    treehop_result.confidence = 0.0  # placeholder

    # -------------------------------------------------
    # Step 4: Routing decision
    # -------------------------------------------------
    # if needs_refinement(treehop_result, threshold=confidence_threshold):
    if treehop_result.confidence < confidence_threshold:

        # ---------------------------------------------
        # Step 4a: RQ-RAG query refinement
        # ---------------------------------------------
        # refiner = QueryRefiner()
        # refined_queries = refiner.refine(
        #     query=query_text,
        #     passages=treehop_result.passages,
        # )

        refined_queries = []  # placeholder

        # ---------------------------------------------
        # Step 4b: Re-run retrieval with refined query
        # ---------------------------------------------
        # query_embedding = embed_query(refined_queries[0])
        # treehop_result = treehop.run(query_embedding)

        pass

    # -------------------------------------------------
    # Step 5: Answer generation
    # -------------------------------------------------
    # generator = AnswerGenerator()
    # answer = generator.generate(query_text, treehop_result.passages)

    answer = "[ANSWER PLACEHOLDER]"

    return answer


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Example question"
    result = run_pipeline(query)
    print("\n=== FINAL ANSWER ===\n")
    print(result)
