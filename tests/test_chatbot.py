"""Tests for Memora's in-session conversation history."""

from copy import deepcopy
from io import StringIO
import builtins
import contextlib
import sys
import types
import unittest
from unittest.mock import patch


try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=object)

import chatbot


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.input_tokens = types.SimpleNamespace(
            count=lambda **kwargs: types.SimpleNamespace(
                input_tokens=len(kwargs["input"]) * 10,
            )
        )
        self.replies = iter(
            ["了解，我會記住。", "你的英文程度是 B1。"]
        )

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        return types.SimpleNamespace(
            output_text=next(self.replies),
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class ChatbotTest(unittest.TestCase):
    def test_history_is_sent_on_the_next_request(self) -> None:
        client = FakeClient()
        inputs = iter(
            [
                "我的英文程度是 B1。",
                "我的英文程度是多少？",
                "history",
                "exit",
            ]
        )
        output = StringIO()

        with (
            patch.object(chatbot, "OpenAI", return_value=client),
            patch.object(builtins, "input", side_effect=lambda _="": next(inputs)),
            contextlib.redirect_stdout(output),
        ):
            chatbot.main()

        first_input = client.responses.calls[0]["input"]
        second_input = client.responses.calls[1]["input"]

        self.assertEqual(
            first_input,
            [{"role": "user", "content": "我的英文程度是 B1。"}],
        )
        self.assertEqual(len(second_input), 3)
        self.assertEqual(second_input[1]["role"], "assistant")
        self.assertIn("assistant: 你的英文程度是 B1。", output.getvalue())
        self.assertIn("Session total tokens: 30", output.getvalue())
        self.assertIn("Bye!", output.getvalue())

    def test_context_window_keeps_complete_recent_turns(self) -> None:
        client = FakeClient()
        history = []
        for index in range(5):
            history.extend(
                [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            )
        history.append({"role": "user", "content": "current question"})

        context, token_count = chatbot.build_context_window(client, history)

        self.assertEqual(len(context), 7)
        self.assertEqual(context[0]["content"], "question 2")
        self.assertEqual(context[-1]["content"], "current question")
        self.assertEqual(token_count, 70)


if __name__ == "__main__":
    unittest.main()
