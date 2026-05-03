import os

from openai import OpenAI

from .base import LanguageModel


class OpenAIModel(LanguageModel):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(
        self, message: str, system_msg: str = None, max_tokens: int = 200, **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_msg or "You are a helpful assistant.",
                },
                {"role": "user", "content": message},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
