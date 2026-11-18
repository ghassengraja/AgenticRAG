from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

def get_model():
    if not settings.VLLM_BASE_URL or not settings.VLLM_MODEL:
        raise ImproperlyConfigured("Set VLLM_BASE_URL and VLLM_MODEL to use vLLM.")

    return ChatOpenAI(
        model=settings.VLLM_MODEL,
        base_url=settings.VLLM_BASE_URL,
        max_tokens=1024,
    )

def get_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",google_api_key=settings.GOOGLE_API_KEY)
    return embedding_model
