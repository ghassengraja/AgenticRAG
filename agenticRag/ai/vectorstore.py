import os
from langchain_chroma import Chroma
from django.conf import settings
from .llm import get_embedding_model

def get_vectorstore():
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_db_episodic_memory")
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name="episodic_memory",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore
