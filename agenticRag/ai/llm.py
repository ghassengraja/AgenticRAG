import json
from urllib.request import Request, urlopen

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI


class VLLMEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            payload = {
                "model": self.model,
                "input": texts[start : start + batch_size],
                "encoding_format": "float",
                "truncate_prompt_tokens": -1,
            }
            request = Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=60) as response:
                data = json.load(response)["data"]
            embeddings.extend(item["embedding"] for item in sorted(data, key=lambda item: item["index"]))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_model():
    if not settings.VLLM_BASE_URL or not settings.VLLM_MODEL:
        raise ImproperlyConfigured("Set VLLM_BASE_URL and VLLM_MODEL to use vLLM.")

    return ChatOpenAI(
        model=settings.VLLM_MODEL,
        base_url=settings.VLLM_BASE_URL,
        max_tokens=1024,
    )


def get_embedding_model():
    if not settings.VLLM_EMBEDDINGS_BASE_URL or not settings.VLLM_EMBEDDINGS_MODEL:
        raise ImproperlyConfigured(
            "Set VLLM_EMBEDDINGS_BASE_URL and VLLM_EMBEDDINGS_MODEL to use vLLM embeddings."
        )

    return VLLMEmbeddings(
        base_url=settings.VLLM_EMBEDDINGS_BASE_URL,
        model=settings.VLLM_EMBEDDINGS_MODEL,
    )
