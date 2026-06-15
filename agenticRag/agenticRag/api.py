import json
import time
from uuid import uuid4
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from ai.supervisor import get_supervisor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

User = get_user_model()

@csrf_exempt
def chat_completions(request):
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Accepts JSON payload with `messages` and maps Authorization token to Django User.
    """
    if request.method != 'POST':
        return JsonResponse({"status": "ok", "message": "Ready for POST requests"}, status=200)
        
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    # 1. Authentication
    # Expecting: Authorization: Bearer <username>
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JsonResponse({"error": "Missing or invalid Authorization header. Use 'Bearer <username>'"}, status=401)
        
    token = auth_header.split(" ")[1]
    
    try:
        user = User.objects.get(username=token)
    except User.DoesNotExist:
        return JsonResponse({"error": f"User with username '{token}' not found"}, status=401)
        
    # 2. Parse Messages
    raw_messages = data.get("messages", [])
    langchain_messages = []
    
    for msg in raw_messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))
            
    if not langchain_messages:
        return JsonResponse({"error": "No messages provided"}, status=400)
        
    # 3. Setup LangGraph Config
    # We use a static thread_id per user so their conversation history is continuous in short term memory
    thread_id = f"user_thread_{user.id}"
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user.id
        }
    }
    
    # 4. Invoke Supervisor
    supervisor = get_supervisor()
    try:
        # LangGraph MessagesState expects a dict with a "messages" key
        result = supervisor.invoke({"messages": langchain_messages}, config=config)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
        
    from django.http import StreamingHttpResponse
    
    # Extract the final AI message
    final_messages = result.get("messages", [])
    if final_messages:
        last_message = final_messages[-1].content
    else:
        last_message = "No response generated."
        
    is_stream = data.get("stream", False)
    
    if is_stream:
        def event_stream():
            chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("model", "agentic-rag-supervisor"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": last_message
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        
    # 5. Format standard non-streaming OpenAI Response
    response_data = {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", "agentic-rag-supervisor"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": last_message
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    
    return JsonResponse(response_data)

@csrf_exempt
def get_models(request):
    """
    OpenAI-compatible /v1/models endpoint.
    Used by frontends (like Open WebUI) to verify connection and list available models.
    """
    response_data = {
        "object": "list",
        "data": [
            {
                "id": "agentic-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "backend"
            }
        ]
    }
    return JsonResponse(response_data)

@csrf_exempt
def v1_health(request):
    """
    Health check for /v1 root endpoint.
    """
    return JsonResponse({"status": "ok", "message": "Agentic RAG API v1 is running"})
