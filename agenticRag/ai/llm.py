from langchain_openrouter import ChatOpenRouter
from django.conf import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_model():
    model = ChatOpenRouter(
        model="google/gemma-4-31b-it:free",
        temperature=0,
        max_tokens=1024,
        max_retries=2,
        api_key=settings.OPENROUTER_API_KEY,
    )
    return model

def get_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",google_api_key=settings.GOOGLE_API_KEY)
    return embedding_model
