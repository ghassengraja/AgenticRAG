import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from django.conf import settings

def get_vectorstore():
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_db")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=settings.GOOGLE_API_KEY
    )
    vectorstore = Chroma(
        collection_name="episodic_memory",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore
