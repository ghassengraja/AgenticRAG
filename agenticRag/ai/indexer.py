import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from django.conf import settings
from .llm import get_embedding_model

def create_vector_store():
    """
    Reads PDFs from the data directory, chunks them, and inserts them into ChromaDB.
    """
    embeddings = get_embedding_model()
    
    pdf_dir = r"d:\pfe_yasmine\data"
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_db_docs")
    collection_name = "docs"
    
    print("Loading PDFs and creating vector store...")
    os.makedirs(persist_directory, exist_ok=True)
    
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
    loader = PyPDFDirectoryLoader(pdf_dir)
    try:
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {pdf_dir}")
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        raise
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    pages_split = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")
    return vectorstore
