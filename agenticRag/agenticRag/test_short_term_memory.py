from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, override_settings
from langgraph.checkpoint.memory import MemorySaver
from redis.exceptions import ConnectionError

from ai.short_term_mem import _get_redis_checkpointer, get_checkpointer


class ShortTermMemoryTests(SimpleTestCase):
    def setUp(self):
        _get_redis_checkpointer.cache_clear()
        self.addCleanup(_get_redis_checkpointer.cache_clear)

    @override_settings(REDIS_URI="")
    @patch("ai.short_term_mem.RedisSaver")
    def test_empty_uri_uses_shared_process_memory(self, redis_saver):
        saver = get_checkpointer()

        self.assertIsInstance(saver, MemorySaver)
        self.assertIs(get_checkpointer(), saver)
        redis_saver.assert_not_called()

    @override_settings(
        REDIS_URI="redis://:test-password@192.0.2.10:6379/0",
        REDIS_CHECKPOINT_TTL_MINUTES=90,
    )
    @patch("ai.short_term_mem.RedisSaver")
    def test_configured_redis_saver_is_initialized_once(self, redis_saver):
        saver = get_checkpointer()

        self.assertIs(saver, redis_saver.return_value)
        self.assertIs(get_checkpointer(), saver)
        redis_saver.assert_called_once_with(
            redis_url="redis://:test-password@192.0.2.10:6379/0",
            connection_args={"socket_connect_timeout": 3, "socket_timeout": 5},
            ttl={"default_ttl": 90, "refresh_on_read": True},
        )
        saver.setup.assert_called_once_with()

    @override_settings(REDIS_URI="redis://127.0.0.1:6379/0")
    @patch("ai.short_term_mem.RedisSaver")
    def test_failed_index_setup_can_be_retried(self, redis_saver):
        saver = redis_saver.return_value
        saver.setup.side_effect = [ConnectionError("Redis unavailable"), None]

        with self.assertRaises(ConnectionError):
            get_checkpointer()

        self.assertIs(get_checkpointer(), saver)
        self.assertEqual(saver.setup.call_count, 2)

    @override_settings(REDIS_URI="redis://127.0.0.1:6379/0", REDIS_CHECKPOINT_TTL_MINUTES=0)
    def test_nonpositive_ttl_is_rejected(self):
        with self.assertRaisesMessage(ImproperlyConfigured, "REDIS_CHECKPOINT_TTL_MINUTES"):
            get_checkpointer()
