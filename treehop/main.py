from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever


EVALUATE_DATASET = "2wiki"

# load TreeHop model from HuggingFace
tree_hop_model = TreeHopModel.from_pretrained("allen-li1231/treehop-rag")

# load retriever
retriever = MultiHopRetriever(
    "BAAI/bge-m3",
    passages=f"embedding_data/{EVALUATE_DATASET}/eval_passages_10.jsonl",
    # passage_embeddings=f"embedding_data/{EVALUATE_DATASET}/eval_content_dense.npy",
    # uncomment this if faiss index is initialized, resulting in a faster loading
    faiss_index=f"embedding_data/{EVALUATE_DATASET}/index.faiss",
    tree_hop_model=tree_hop_model,
    projection_size=1024,
    save_or_load_index=True,
    indexing_batch_size=10240,
    index_device="cuda"     # or cpu on Apple Metal
)

# employ networkx graph to depict multi-hop retrieval
retrieve_result = retriever.multihop_search_passages(
    "Who is the mother of the director of film Polish-Russian War (Film)?",
    n_hop=2,
    top_n=5,
    index_batch_size=2048,
    generate_batch_size=1024,
    # return_tree=True        # simply add this argument
)

# retrieved passages for questions
print(retrieve_result.passage)

# # `retrieve_result.tree_hop_graph` is a list of networkx objects
# # correspondent to the retrieval paths of the queries in LIST_OF_QUESTIONS.
# # take the first query for example, to draw the respective path:
# retrieval_tree = retrieve_result.tree_hop_graph[0]
# retrieval_tree.plot_tree()

# # nodes represent passages in the retrieval graph
# # store metadata for the original passages:
# print(retrieval_tree.nodes(data=True))