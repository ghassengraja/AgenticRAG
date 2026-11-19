from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, SimpleTestCase, TestCase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class SessionChatTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="reader", password="test-password"
        )

    def setUp(self):
        self.client = Client(enforce_csrf_checks=True)
        self.chat_payload = {"messages": [{"role": "user", "content": "Hello"}]}

    @patch("agenticRag.api.get_supervisor")
    def test_chat_uses_the_signed_in_user_and_requires_csrf(self, get_supervisor):
        token = self.client.get("/v1/session").json()["csrf_token"]
        login = self.client.post(
            "/v1/session",
            {"username": self.user.username, "password": "test-password"},
            content_type="application/json",
            headers={"X-CSRFToken": token},
        )
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json()["authenticated"])

        rejected = self.client.post(
            "/v1/chat/completions", self.chat_payload, content_type="application/json",
            headers={"X-CSRFToken": token},
        )
        self.assertEqual(rejected.status_code, 403)
        get_supervisor.assert_not_called()

        invoke = get_supervisor.return_value.invoke
        invoke.return_value = {"messages": [AIMessage(content="Hi")]}
        headers = {"X-CSRFToken": login.json()["csrf_token"]}
        response = self.client.post(
            "/v1/chat/completions", self.chat_payload, content_type="application/json",
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "Hi")
        self.assertNotIn("usage", response.json())
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.args, ({"messages": [HumanMessage(content="Hello")]},))
        self.assertEqual(invoke.call_args.kwargs["config"]["configurable"], {
            "thread_id": f"user_thread_{self.user.id}",
            "user_id": self.user.id,
        })

        signout = self.client.delete("/v1/session", headers=headers)
        self.assertEqual(signout.status_code, 200)
        self.assertFalse(signout.json()["authenticated"])
        response = self.client.post(
            "/v1/chat/completions", self.chat_payload, content_type="application/json",
            headers={"X-CSRFToken": signout.json()["csrf_token"]},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(invoke.call_count, 1)

    def test_signin_rejects_missing_csrf_and_invalid_credentials(self):
        credentials = {"username": self.user.username, "password": "wrong-password"}
        token = self.client.get("/v1/session").json()["csrf_token"]
        response = self.client.post(
            "/v1/session", credentials, content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)
        response = self.client.post(
            "/v1/session", credentials, content_type="application/json",
            headers={"X-CSRFToken": token},
        )
        self.assertEqual(response.status_code, 401)
        self.assertFalse(self.client.get("/v1/session").json()["authenticated"])


class ChatContractTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(username="reader")

    def setUp(self):
        self.client.force_login(self.user)

    @patch("agenticRag.api.get_supervisor")
    def test_invalid_or_unsupported_requests_do_not_run_the_graph(self, get_supervisor):
        messages = [{"role": "user", "content": "Hello"}]
        cases = [
            ("[]", 400, None),
            ({"messages": []}, 400, "messages"),
            ({"messages": [None]}, 400, "messages[0]"),
            ({"messages": [{"role": "tool", "content": "Hello"}]}, 400, "messages[0].role"),
            ({"messages": [{"role": "user", "content": None}]}, 400, "messages[0].content"),
            ({"messages": messages, "model": "unknown"}, 404, "model"),
            ({"messages": messages, "stream": "false"}, 400, "stream"),
            ({"messages": messages, "stream": True}, 400, "stream"),
            ({"messages": messages, "temperature": 0}, 400, "temperature"),
        ]
        for payload, status, param in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/v1/chat/completions", payload, content_type="application/json")
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["param"], param)
        get_supervisor.assert_not_called()

    @patch("agenticRag.api.get_supervisor")
    def test_chat_reports_public_model_and_usage(self, get_supervisor):
        parts = [{"type": "text", "text": "Hello"}]
        model = self.client.get("/v1/models").json()["data"][0]["id"]

        def invoke(state, *, config):
            self.assertEqual(state["messages"], [
                SystemMessage(content="Be brief", additional_kwargs={"__openai_role__": "developer"}),
                HumanMessage(content=parts),
            ])
            callback, = config["callbacks"]
            for input_tokens, output_tokens in [(4, 2), (6, 3)]:
                answer = AIMessage(
                    content="Hi",
                    response_metadata={"model_name": "upstream-model", "finish_reason": "length"},
                    usage_metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                )
                callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=answer)]]))
            return {"messages": [answer]}

        get_supervisor.return_value.invoke.side_effect = invoke
        response = self.client.post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "developer", "content": "Be brief"}, {"role": "user", "content": parts}],
        }, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "agentic-rag")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
        self.assertEqual(response.json()["usage"], {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        })


class APIRouteTests(SimpleTestCase):
    def test_api_routes_enforce_http_methods(self):
        chat = self.client.get("/v1/chat/completions")
        self.assertEqual(chat.status_code, 405)
        self.assertEqual(chat["Allow"], "POST")
        self.assertEqual(self.client.post("/v1/models").status_code, 405)
        self.assertEqual(self.client.get("/v1/models").status_code, 200)

    def test_health_uses_the_trailing_slash_route(self):
        self.assertRedirects(self.client.get("/v1"), "/v1/", status_code=301)
        self.assertEqual(self.client.get("/v1/").json()["status"], "ok")
