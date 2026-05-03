"""
Adaptive RAG pipeline.

Routes questions to TreeHop+Reader (fast) or RQ-RAG (accurate) based on type:
  comparison, inference       → TreeHop + LLM reader
  compositional, bridge_comparison → RQ-RAG (fine-tuned Llama)
"""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
_treehop = os.path.join(_root, "treehop")
_rqrag_lm = os.path.join(_root, "rq-rag", "retrieval_lm")
_rqrag = os.path.join(_root, "rq-rag")

# TreeHop imports first so its local modules cache under their real names.
sys.path.insert(0, _treehop)
from tree_hop import TreeHopModel
from passage_retrieval import MultiHopRetriever
from reader import Reader

# Remove treehop from sys.path and evict every module name that conflicts with
# rq-rag/retrieval_lm/ (utils, metrics, f1, evaluation) before loading rq-rag.
sys.path.remove(_treehop)
for _mod in ("utils", "metrics", "f1", "evaluation"):
    sys.modules.pop(_mod, None)

sys.path.insert(0, _rqrag)
sys.path.insert(0, _rqrag_lm)
from utils import load_sag_special_tokens, preprocess_eval_data
from inference import generate_tree_of_thoughts
from data_curation.tools.bm25_candidates import BM25Run

import torch
from argparse import Namespace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

TREEHOP_LABELS = {"comparison", "inference"}
RQRAG_LABELS = {"compositional", "bridge_comparison"}


class RouterClassifier:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label  # {"0": "comparison", ...}

    def classify(self, question: str) -> str:
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return self.id2label[logits.argmax(dim=-1).item()]


class RQRAGPipeline:
    def __init__(
        self,
        model_name_or_path: str,
        ndocs: int = 3,
        max_depth: int = 2,
        task: str = "2wikimultihopqa",
    ):
        self.task = task
        self.max_depth = max_depth

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.special_tokens_dict = load_sag_special_tokens(self.tokenizer)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_4bit=True, device_map="auto"
        )
        self.model.generation_config.eos_token_id = [
            self.tokenizer.convert_tokens_to_ids("</s>"),
            self.tokenizer.convert_tokens_to_ids("[EOS]"),
        ]
        self.model.generation_config.max_new_tokens = 100

        self.search_engine = BM25Run(ndocs)
        self.args = Namespace(
            task=task,
            search_engine_type="bm25_candidates",
            expand_on_tokens=[
                "[S_Rewritten_Query]",
                "[S_Decomposed_Query]",
                "[S_Disambiguated_Query]",
                "[A_Response]",
            ],
            oracle=False,
        )

    def answer(self, question: str, context: list = None) -> str:
        row = {"question": question, "context": context or [], "answers": []}
        eval_data = preprocess_eval_data(
            [row], tokenizer=self.tokenizer, task=self.task
        )
        preds, _ = generate_tree_of_thoughts(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_prompts=eval_data,
            raw_datas=[row],
            special_tokens_dict=self.special_tokens_dict,
            max_depth=self.max_depth,
            max_width=3,
            search_engine_api=self.search_engine,
            search_limit=1,
            args=self.args,
            index=0,
            total_corpus=None,
        )
        return preds[0].strip()


class AdaptiveRAGPipeline:
    def __init__(
        self,
        router_path: str,
        treehop_retriever: MultiHopRetriever,
        reader: Reader,
        rqrag_pipeline: RQRAGPipeline = None,
        n_hop: int = 2,
        top_n: int = 5,
    ):
        self.router = RouterClassifier(router_path)
        self.retriever = treehop_retriever
        self.reader = reader
        self.rqrag = rqrag_pipeline
        self.n_hop = n_hop
        self.top_n = top_n

    def answer(self, question: str, context: list = None) -> dict:
        label = self.router.classify(question)

        if label in TREEHOP_LABELS:
            retrieve_result = self.retriever.multihop_search_passages(
                question, n_hop=self.n_hop, top_n=self.top_n
            )
            ans = self.reader.answer(question, retrieve_result)
            path = "treehop"
        else:
            if self.rqrag is None:
                raise RuntimeError(
                    f"Question type '{label}' routes to RQ-RAG but no rqrag_pipeline "
                    "was provided. Pass an RQRAGPipeline instance to AdaptiveRAGPipeline."
                )
            ans = self.rqrag.answer(question, context=context)
            path = "rqrag"

        return {"question": question, "type": label, "path": path, "answer": ans}
