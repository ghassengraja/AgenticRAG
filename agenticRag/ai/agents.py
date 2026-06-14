from django.conf import settings
from .llm import get_model,get_embedding_model
from langgraph.prebuilt import create_react_agent
from tools_documents import document_tools
from tools_memory import memory_tools
from tools_retriever import retriever_tool
from langchain_core.messages import SystemMessage
from short_term_mem import saver as checkpointer

def get_document_agent():
    model = get_model()
    agent = create_react_agent(
        model=model,
        tools=document_tools,
        prompt="You are a polite helpful assistant for managing documents.",
        checkpointer=checkpointer
    )
    return agent


def get_memory_agent(checkpointer=None):
    model = get_model()
    
    agent_prompt = """You are a specialized memory agent responsible for maintaining the user's long-term memory bank. 
When instructed to save new facts from the recent conversation, you MUST follow this strict deduplication workflow:
1. Read the recent conversation to extract key facts.
2. ALWAYS use `search_episodic_memory` to check if similar facts already exist.
3. If a fact already exists but is outdated, use `update_episodic_memory` using its ID.
4. If a fact directly contradicts the new truth and shouldn't just be updated, use `delete_episodic_memory`.
5. Only if the fact is completely new, use `add_episodic_memory`."""

    agent = create_react_agent(
        model=model,
        tools=memory_tools,
        prompt=agent_prompt,
        checkpointer=checkpointer
    )
    return agent

def rag_document_agent():
    model = get_model() # Must use Chat LLM, not embeddings model
    agent = create_react_agent(
        model=model,
        tools=[retriever_tool],
        prompt="You are a helpful assistant that answers questions using the retriever tool.",
        checkpointer=checkpointer
    )
    return agent