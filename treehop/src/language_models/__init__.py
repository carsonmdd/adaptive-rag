from .aoai import AOAI
from .base import LanguageModel
from .deepseek import DeepSeek
from .llama import LlamaServer
from .openai_model import OpenAIModel
from .utils import ask_model, ask_model_in_parallel

MODEL_DICT = {
    "gpt35": "gpt-35-turbo-1106",
    "gpt4": "gpt-4-0125-preview",
    "gpt4o-mini": "gpt-4o-mini",
    "llama3-70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-chat",
}