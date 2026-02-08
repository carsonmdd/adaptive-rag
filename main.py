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


def main():
    LIST_OF_QUESTIONS = [
        "Who is the mother of the director of film Polish-Russian War (Film)?"
    ]

    # employ networkx graph to depict multi-hop retrieval
    retrieve_result = retriever.multihop_search_passages(
        LIST_OF_QUESTIONS,
        n_hop=2,
        top_n=5,
        index_batch_size=2048,
        generate_batch_size=1024,
        return_tree=True        # simply add this argument
    )

    # retrieved passages for questions
    print(retrieve_result.passage)

    # `retrieve_result.tree_hop_graph` is a list of networkx objects
    # correspondent to the retrieval paths of the queries in LIST_OF_QUESTIONS.
    # take the first query for example, to draw the respective path:
    retrieval_tree = retrieve_result.tree_hop_graph[0]
    retrieval_tree.plot_tree()

    # nodes represent passages in the retrieval graph
    # store metadata for the original passages:
    print(retrieval_tree.nodes(data=True))


if __name__ == "__main__":
    # query = sys.argv[1] if len(sys.argv) > 1 else "Example question"
    # result = run_pipeline(query)
    # print("\n=== FINAL ANSWER ===\n")
    # print(result)
    main()