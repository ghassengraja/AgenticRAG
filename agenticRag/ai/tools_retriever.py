from langchain_core.tools import tool
import os
from langchain_chroma import Chroma
from django.conf import settings
from .llm import get_embedding_model





def get_retriever():
    embeddings = get_embedding_model()
    
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_db_docs")
    collection_name = "docs"
    
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector store not found. Please run the document indexer to create it.")
        
    # Load existing vectorstore
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
        
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the documents.
    """

    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the documents."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]
