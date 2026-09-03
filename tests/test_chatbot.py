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
        self.extraction_result = chatbot.MemoryExtractionResult(
            should_remember=False,
            memories=[],
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

    def parse(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        return types.SimpleNamespace(
            output_parsed=self.extraction_result,
            usage=types.SimpleNamespace(
                input_tokens=8,
                output_tokens=2,
                total_tokens=10,
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

        chat_calls = [
            call
            for call in client.responses.calls
            if call.get("instructions") == chatbot.SYSTEM_PROMPT
        ]
        first_input = chat_calls[0]["input"]
        second_input = chat_calls[1]["input"]

        self.assertEqual(
            first_input,
            [{"role": "user", "content": "我的英文程度是 B1。"}],
        )
        self.assertEqual(len(second_input), 3)
        self.assertEqual(second_input[1]["role"], "assistant")
        self.assertIn("assistant: 你的英文程度是 B1。", output.getvalue())
        self.assertIn("Session total tokens: 50", output.getvalue())
        self.assertIn("Bye!", output.getvalue())

    def test_old_turns_are_summarized_and_recent_turns_stay_verbatim(self) -> None:
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

        memory = chatbot.ShortTermMemory(client=client)
        memory.history = history
        stats = memory.prepare_context()

        self.assertEqual(stats["newly_summarized_count"], 4)
        self.assertEqual(memory.summarized_message_count, 4)
        self.assertEqual(stats["context_messages"][0]["role"], "developer")
        self.assertEqual(stats["context_messages"][1]["content"], "question 2")
        self.assertEqual(stats["context_messages"][-1]["content"], "current question")

    def test_failed_request_can_roll_back_the_pending_user_message(self) -> None:
        memory = chatbot.ShortTermMemory(client=FakeClient())
        memory.add_user_message("temporary")
        memory.rollback_last_user_message()
        self.assertEqual(memory.history, [])

    def test_remember_command_saves_a_candidate(self) -> None:
        client = FakeClient()
        inputs = iter(["remember 我的英文程度是 B1。", "memories", "exit"])
        output = StringIO()

        with (
            patch.object(chatbot, "OpenAI", return_value=client),
            patch.object(builtins, "input", side_effect=lambda _="": next(inputs)),
            contextlib.redirect_stdout(output),
        ):
            chatbot.main()

        self.assertIn("Saved as Memory Candidate", output.getvalue())
        self.assertIn("1. 我的英文程度是 B1。", output.getvalue())

    def test_extracts_structured_memory_candidates(self) -> None:
        client = FakeClient()
        client.responses.extraction_result = chatbot.MemoryExtractionResult(
            should_remember=True,
            memories=[
                chatbot.MemoryCandidate(content="使用者的英文程度是 B1。"),
                chatbot.MemoryCandidate(content="使用者想加強旅遊英文。"),
            ],
        )

        result, tokens = chatbot.extract_memory_candidates(
            client,
            "我的英文程度是 B1，我想加強旅遊英文。",
        )

        self.assertTrue(result.should_remember)
        self.assertEqual(len(result.memories), 2)
        self.assertEqual(result.memories[0].content, "使用者的英文程度是 B1。")
        self.assertEqual(tokens, 10)


if __name__ == "__main__":
    unittest.main()
