from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from ai.vectorstore import get_vectorstore
from documents.models import EpisodicMemory
from django.contrib.auth import get_user_model

User = get_user_model()


@tool
def search_episodic_memory(query: str, limit: int = 5, config: RunnableConfig = None):
    """
    Search the episodic memory vector database to recall past experiences or facts.
    
    Arguments:
    query: The search query to recall relevant memories.
    limit: The maximum number of results to return (default 5, max 25).
    """
    if limit > 25:
        limit = 25
        
    config = config or {}
    configurable = config.get('configurable') or config.get('metadata') or {}
    user_id = configurable.get('user_id')
    
    if user_id is None:
        return "Invalid request: missing user_id."
        
    vectorstore = get_vectorstore()
    
    filter_dict = {
        "owner_id": str(user_id)
    }
    
    results = vectorstore.similarity_search(
        query=query,
        k=limit,
        filter=filter_dict
    )
    
    response_data = []
    for doc in results:
        response_data.append({
            "id": doc.metadata.get("id"),
            "memory": doc.page_content,
        })
        
    return response_data

@tool
def add_episodic_memory(fact: str, context: str, config: RunnableConfig = None):
    """
    Save a completely new fact or experience to the agent's dynamic episodic memory.
    Use this ONLY after searching to confirm the fact does not already exist.
    
    Arguments:
    fact: A concise statement of the fact or experience.
    context: Detailed context or explanation surrounding the fact.
    """
    config = config or {}
    configurable = config.get('configurable') or config.get('metadata') or {}
    user_id = configurable.get('user_id')
    
    if user_id is None:
        return "Invalid request: missing user_id."
        
    try:
        user = User.objects.get(id=user_id)
        memory = EpisodicMemory.objects.create(
            owner=user,
            fact=fact,
            context=context
        )
        
        vectorstore = get_vectorstore()
        vectorstore.add_texts(
            texts=[fact],
            metadatas=[{"id": str(memory.id), "owner_id": str(user_id)}],
            ids=[str(memory.id)]
        )
        
        return f"Successfully saved NEW memory ID {memory.id}."
    except Exception as e:
        return f"Failed to save memory: {str(e)}"

@tool
def update_episodic_memory(memory_id: str, fact: str, context: str, config: RunnableConfig = None):
    """
    Update an existing memory with new information.
    Use this when a previously saved fact has changed or needs more context.
    
    Arguments:
    memory_id: The UUID string of the memory to update (obtained from search_episodic_memory).
    fact: The updated concise statement of the fact.
    context: The updated detailed context.
    """
    config = config or {}
    configurable = config.get('configurable') or config.get('metadata') or {}
    user_id = configurable.get('user_id')
    
    if user_id is None:
        return "Invalid request: missing user_id."
        
    try:
        memory = EpisodicMemory.objects.get(id=memory_id, owner_id=user_id)
        memory.fact = fact
        memory.context = context
        memory.save()
        
        vectorstore = get_vectorstore()
        try:
            vectorstore.delete(ids=[str(memory.id)])
        except Exception:
            pass
        vectorstore.add_texts(
            texts=[fact],
            metadatas=[{"id": str(memory.id), "owner_id": str(user_id)}],
            ids=[str(memory.id)]
        )
        
        return f"Successfully UPDATED memory ID {memory.id}."
    except EpisodicMemory.DoesNotExist:
        return f"Memory ID {memory_id} not found."
    except Exception as e:
        return f"Failed to update memory: {str(e)}"

@tool
def delete_episodic_memory(memory_id: str, config: RunnableConfig = None):
    """
    Delete an existing memory.
    Use this when a previously saved fact is no longer true, directly contradicted, or completely irrelevant.
    
    Arguments:
    memory_id: The UUID string of the memory to delete (obtained from search_episodic_memory).
    """
    config = config or {}
    configurable = config.get('configurable') or config.get('metadata') or {}
    user_id = configurable.get('user_id')
    
    if user_id is None:
        return "Invalid request: missing user_id."
        
    try:
        memory = EpisodicMemory.objects.get(id=memory_id, owner_id=user_id)
        memory.delete()
        
        vectorstore = get_vectorstore()
        try:
            vectorstore.delete(ids=[str(memory_id)])
        except Exception:
            pass
            
        return f"Successfully DELETED memory ID {memory_id}."
    except EpisodicMemory.DoesNotExist:
        return f"Memory ID {memory_id} not found."
    except Exception as e:
        return f"Failed to delete memory: {str(e)}"

memory_tools = [
    search_episodic_memory,
    add_episodic_memory,
    update_episodic_memory,
    delete_episodic_memory
]