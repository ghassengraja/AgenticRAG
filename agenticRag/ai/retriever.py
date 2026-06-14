import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from django.conf import settings
from .llm import get_embedding_model

def get_stock_retriever():
    embeddings = get_embedding_model()
    
    pdf_dir = r"d:\pfe_yasmine\data"
    
    persist_directory = os.path.join(settings.BASE_DIR, "chroma_db_stock")
    collection_name = "stock_market"
    
    # Check if the DB already exists by checking if the persist directory exists
    if not os.path.exists(persist_directory):
        print("Vector store not found. Loading PDFs and creating it...")
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
    else:
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
