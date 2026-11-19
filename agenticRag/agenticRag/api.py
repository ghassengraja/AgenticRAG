import json
import logging
import time
from uuid import uuid4

from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods, require_POST, require_safe
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, convert_to_messages

from ai.supervisor import get_supervisor


logger = logging.getLogger(__name__)
MODEL_ID = "agentic-rag"
MODEL_CREATED = 1781520976  # The first commit that advertised this model ID.
CHAT_ROLES = ("user", "assistant", "system", "developer")
CHAT_FIELDS = {"model", "messages", "stream"}


def chat_error(message, status=400, *, param=None, code=None, error_type="invalid_request_error"):
    return JsonResponse({"error": {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }}, status=status)


def is_text_content(content):
    if isinstance(content, str):
        return True
    return isinstance(content, list) and bool(content) and all(
        isinstance(part, dict)
        and part.keys() == {"type", "text"}
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        for part in content
    )


@never_cache
@require_http_methods(["GET", "POST", "DELETE"])
def session(request):
    """Read, create, or end a Django login session."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        if not isinstance(data, dict):
            return JsonResponse({"error": "Expected a JSON object"}, status=400)

        username = data.get("username")
        password = data.get("password")
        if not isinstance(username, str) or not isinstance(password, str) or not username or not password:
            return JsonResponse({"error": "Username and password are required"}, status=400)

        user = authenticate(request, username=username, password=password)
        if user is None:
            return JsonResponse({"error": "Invalid username or password"}, status=401)
        login(request, user)
    elif request.method == "DELETE":
        logout(request)

    return JsonResponse({
        "authenticated": request.user.is_authenticated,
        "csrf_token": get_token(request),
    })


@require_POST
def chat_completions(request):
    """Return a text chat completion for a signed-in Django user."""
    user = request.user
    if not user.is_authenticated:
        return chat_error("Sign in to use chat completions", 401, error_type="authentication_error")

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return chat_error("Invalid JSON")
    if not isinstance(data, dict):
        return chat_error("Expected a JSON object")

    unsupported_field = next((field for field in data if field not in CHAT_FIELDS), None)
    if unsupported_field is not None:
        return chat_error("Unsupported chat parameter", param=unsupported_field)

    model = data.get("model", MODEL_ID)
    if not isinstance(model, str):
        return chat_error("Model must be a string", param="model")
    if model != MODEL_ID:
        return chat_error(
            f"Available model is '{MODEL_ID}'", 404, param="model", code="model_not_found"
        )

    stream = data.get("stream")
    if stream is not None and not isinstance(stream, bool):
        return chat_error("Stream must be a boolean", param="stream")
    if stream:
        return chat_error("Streaming is unavailable. Send stream=false or omit it.", param="stream")

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return chat_error("Messages must be a non-empty array", param="messages")

    messages = []
    for index, message in enumerate(raw_messages):
        param = f"messages[{index}]"
        if not isinstance(message, dict):
            return chat_error("Expected a message object", param=param)
        role = message.get("role")
        if role not in CHAT_ROLES:
            return chat_error(
                "Supported roles are user, assistant, system, and developer", param=f"{param}.role"
            )
        content = message.get("content")
        if not is_text_content(content):
            return chat_error("Content must be text or an array of text parts", param=f"{param}.content")
        messages.append({"role": role, "content": content})

    usage = UsageMetadataCallbackHandler()
    config = {
        "configurable": {
            "thread_id": f"user_thread_{user.id}",
            "user_id": user.id,
        },
        "callbacks": [usage],
    }
    try:
        supervisor = get_supervisor()
        result = supervisor.invoke({"messages": convert_to_messages(messages)}, config=config)
    except Exception:
        logger.exception("Chat completion failed")
        return chat_error("Chat completion failed", 500, error_type="server_error")

    final_messages = result.get("messages", [])
    answer = final_messages[-1] if final_messages else None
    if not isinstance(answer, AIMessage) or answer.tool_calls or answer.invalid_tool_calls:
        return chat_error("Supervisor did not return a final answer", 502, error_type="server_error")

    finish_reason = answer.response_metadata.get("finish_reason")
    if finish_reason not in ("stop", "length", "content_filter"):
        finish_reason = "stop"

    response_data = {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer.text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage.usage_metadata:
        counts = usage.usage_metadata.values()
        response_data["usage"] = {
            "prompt_tokens": sum(count["input_tokens"] for count in counts),
            "completion_tokens": sum(count["output_tokens"] for count in counts),
            "total_tokens": sum(count["total_tokens"] for count in counts),
        }

    return JsonResponse(response_data)


@require_safe
def get_models(request):
    """Return the public model ID used by chat completions."""
    return JsonResponse({
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "backend",
            }
        ],
    })


@require_safe
def v1_health(request):
    """Return the Django API's health status."""
    return JsonResponse({"status": "ok", "message": "Agentic RAG API v1 is running"})
