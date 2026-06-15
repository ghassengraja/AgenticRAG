from langgraph_supervisor import create_supervisor
from ai.llm import get_model
from ai.agents import get_document_agent, get_memory_agent, rag_document_agent
from langsmith import traceable
from .short_term_mem import saver as checkpointer

def get_supervisor(model = None, checkpointer=checkpointer):
    model = get_model()
    document_agent = get_document_agent(checkpointer=checkpointer)
    memory_agent = get_memory_agent(checkpointer=checkpointer)
    rag_agent = rag_document_agent(checkpointer=checkpointer)
    
    document_agent.name = "document_agent"
    memory_agent.name = "memory_agent"
    rag_agent.name = "rag_agent"
    
    supervisor_builder = create_supervisor(
        agents=[
            document_agent,
            memory_agent,
            rag_agent
        ],
        model=model,
        prompt="""You are a supervisor agent that interacts with the user and routes between 
        the document agent, the memory agent, and the RAG agent.
        Use the document agent to manage static user-uploaded documents.
        Use the RAG agent to answer specific queries from the provided PDFs.
        Use the memory agent to recall user preferences to personalize your answers.
        When the session is closing or when the conversation contains significant new facts or preferences, you must route to the memory_agent and instruct it to summarize the recent conversation and save it to long-term episodic memory."""
    )
    
    return supervisor_builder.compile(checkpointer=checkpointer)