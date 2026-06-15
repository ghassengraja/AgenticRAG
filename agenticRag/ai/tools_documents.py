from importlib import metadata
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from documents.models import Document
import os
import logging
from permit import Permit
from django.conf import settings
from asgiref.sync import async_to_sync


def _get_user_document(document_id, user_id, active_only=True):
    """
    Helper: fetch a document if the user is the owner OR a collaborator.
    """
    lookups = {"id": document_id}
    if active_only:
        lookups["active"] = True
    try:
        obj = Document.objects.get(**lookups)
    except Document.DoesNotExist:
        return None
    # Check if user is owner
    if str(obj.owner_id) == str(user_id):
        return obj
    return None

@tool
def get_document(id: str,config:RunnableConfig):
    """Retrieve a specific document by its ID."""
    configurable = config.get('configurable') or config.get('metadata')
    user_id = configurable.get('user_id')
    if user_id is None:
        return "Invalid request for user"
    has_perms = async_to_sync(permit.check)(f"{user_id}", "read", "document")
    if not has_perms:
        raise Exception("You do not have permission to read documents.")
    obj = _get_user_document(id, user_id)
    if obj is None:
        return {"error":"Document not found"}
    return {
        "content": obj.content,
    }


@tool
def search_query_documents(query: str, limit:int=5, config:RunnableConfig = {}):
    """
    Search documents the current user owns or collaborates on, with maximum of 25.

    arguments:
    query: string or lookup search across type, content, topic, received_from, or sent_to
    limit: number of results
    """
    if limit > 25:
        limit = 25
    configurable = config.get('configurable') or config.get('metadata')
    user_id = configurable.get('user_id')
    if user_id is None:
        raise Exception("Invalid request for user.")
    has_perms = async_to_sync(permit.check)(f"{user_id}", "read", "document")
    if not has_perms:
        raise Exception("You do not have permission to search documents.")
    # Return documents where user is owner
    qs = Document.objects.filter(
        owner_id=user_id,
        active=True
    ).filter(
        Q(content__icontains=query) | Q(title__icontains=query)
    ).distinct().order_by("-created_at")
    response_data = []
    for obj in qs[:limit]:
        response_data.append(
            {
                "id": obj.id,
                "title": obj.title
            }
        )
    return response_data


document_tools=[
    search_query_documents,
    get_document
    ]