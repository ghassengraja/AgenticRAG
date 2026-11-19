from functools import cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver


_memory_saver = MemorySaver()


@cache
def _get_redis_checkpointer(uri, ttl_minutes):
    if ttl_minutes < 1:
        raise ImproperlyConfigured("REDIS_CHECKPOINT_TTL_MINUTES must be a positive integer.")

    checkpointer = RedisSaver(
        redis_url=uri,
        connection_args={"socket_connect_timeout": 3, "socket_timeout": 5},
        ttl={"default_ttl": ttl_minutes, "refresh_on_read": True},
    )
    checkpointer.setup()
    return checkpointer


def get_checkpointer():
    """Return the shared LangGraph checkpointer for this Django process."""
    uri = settings.REDIS_URI.strip()
    if not uri:
        return _memory_saver
    return _get_redis_checkpointer(uri, settings.REDIS_CHECKPOINT_TTL_MINUTES)
