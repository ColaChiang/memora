"""Tests for Memora's memory and agent behavior."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO
import builtins
import contextlib
import json
import sys
import tempfile
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
        self.create_results: list[object] = []
        self.extraction_result = chatbot.MemoryExtractionResult(
            should_remember=False,
            memories=[],
        )
        self.importance_result = chatbot.ImportanceAssessmentResult(
            assessments=[
                chatbot.ImportanceAssessment(
                    candidate_index=0,
                    importance_score=3,
                )
            ]
        )
        self.reconciliation_result = chatbot.MemoryReconciliationDecision(
            action="create",
            target_memory_id=None,
            reason="The incoming memory is a separate fact.",
        )

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        if self.create_results:
            return self.create_results.pop(0)
        return types.SimpleNamespace(
            output_text=next(self.replies),
            output=[],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )

    def parse(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        if kwargs.get("text_format") is chatbot.ImportanceAssessmentResult:
            parsed_output = self.importance_result
        elif kwargs.get("text_format") is chatbot.MemoryReconciliationDecision:
            parsed_output = self.reconciliation_result
        else:
            parsed_output = self.extraction_result
        return types.SimpleNamespace(
            output_parsed=parsed_output,
            usage=types.SimpleNamespace(
                input_tokens=8,
                output_tokens=2,
                total_tokens=10,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.embeddings = FakeEmbeddings()


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        texts = kwargs["input"]
        return types.SimpleNamespace(
            data=[
                types.SimpleNamespace(index=index, embedding=[1.0, float(index)])
                for index, _ in enumerate(texts)
            ],
            usage=types.SimpleNamespace(total_tokens=len(texts) * 3),
        )


class FakeCollection:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def add(self, *, ids, documents, embeddings, metadatas) -> None:
        self.records.extend(
            {
                "id": memory_id,
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }
            for memory_id, document, embedding, metadata in zip(
                ids, documents, embeddings, metadatas
            )
        )

    def count(self) -> int:
        return len(self.records)

    def get(self, *, include, ids=None) -> dict[str, object]:
        records = self.records
        if ids is not None:
            records_by_id = {record["id"]: record for record in self.records}
            records = [
                records_by_id[memory_id]
                for memory_id in ids
                if memory_id in records_by_id
            ]
        return {
            "ids": [record["id"] for record in records],
            "documents": [record["document"] for record in records],
            "metadatas": [record["metadata"] for record in records],
        }

    def update(self, *, ids, metadatas, documents=None, embeddings=None) -> None:
        for index, (memory_id, metadata) in enumerate(zip(ids, metadatas)):
            record = next(item for item in self.records if item["id"] == memory_id)
            if documents is not None:
                record["document"] = documents[index]
            if embeddings is not None:
                record["embedding"] = embeddings[index]
            record["metadata"] = metadata

    def delete(self, *, ids) -> None:
        ids_to_delete = set(ids)
        self.records = [
            record for record in self.records if record["id"] not in ids_to_delete
        ]

    def query(self, *, query_embeddings, n_results, include) -> dict[str, object]:
        records = self.records[:n_results]
        distances = [0.1 + index * 0.5 for index, _ in enumerate(records)]
        return {
            "ids": [[record["id"] for record in records]],
            "documents": [[record["document"] for record in records]],
            "metadatas": [[record["metadata"] for record in records]],
            "distances": [distances],
        }


def fake_memory_store() -> chatbot.LongTermMemoryStore:
    return chatbot.LongTermMemoryStore("", "", FakeCollection())


def fake_model_response(
    output_text: str,
    output: list[object],
    total_tokens: int,
) -> object:
    return types.SimpleNamespace(
        output_text=output_text,
        output=output,
        usage=types.SimpleNamespace(
            input_tokens=total_tokens - 2,
            output_tokens=2,
            total_tokens=total_tokens,
        ),
    )


class ChatbotTest(unittest.TestCase):
    def test_history_is_sent_on_the_next_request(self) -> None:
        client = FakeClient()
        inputs = iter(
            [
                "我的英文程度是 B1。",
                "我的英文程度是多少？",
                "status",
                "history",
                "exit",
            ]
        )
        output = StringIO()

        with (
            patch.object(chatbot, "validate_startup"),
            patch.object(chatbot, "OpenAI", return_value=client),
            patch.object(
                chatbot,
                "create_long_term_memory_store",
                return_value=chatbot.LongTermMemoryStore("", "", FakeCollection()),
            ),
            patch.object(
                chatbot,
                "create_user_profile_store",
                return_value=chatbot.UserProfileStore(chatbot.Path("missing-profile.json")),
            ),
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
        self.assertIn("Session total tokens: 30", output.getvalue())
        self.assertIn("Bye!", output.getvalue())

    def test_regular_chat_does_not_run_automatic_retrieval_or_extraction(self) -> None:
        client = FakeClient()
        inputs = iter(["Explain present perfect.", "exit"])

        with (
            patch.object(chatbot, "validate_startup"),
            patch.object(chatbot, "OpenAI", return_value=client),
            patch.object(
                chatbot,
                "create_long_term_memory_store",
                return_value=fake_memory_store(),
            ),
            patch.object(
                chatbot,
                "create_user_profile_store",
                return_value=chatbot.UserProfileStore(chatbot.Path("missing-profile.json")),
            ),
            patch.object(chatbot, "retrieve_relevant_memories") as retrieval,
            patch.object(chatbot, "extract_memory_candidates") as extraction,
            patch.object(builtins, "input", side_effect=lambda _="": next(inputs)),
            contextlib.redirect_stdout(StringIO()),
        ):
            chatbot.main()

        retrieval.assert_not_called()
        extraction.assert_not_called()

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

    def test_count_english_words_treats_contractions_as_one_word(self) -> None:
        result = chatbot.count_english_words("I don't know the answer.")

        self.assertEqual(result["word_count"], 5)
        self.assertIn("contractions", result["counting_rule"])

    def test_count_english_words_validates_tool_input(self) -> None:
        with self.assertRaises(ValueError):
            chatbot.count_english_words(123)
        with self.assertRaises(ValueError):
            chatbot.count_english_words("   ")
        with self.assertRaises(ValueError):
            chatbot.count_english_words("a" * (chatbot.COUNT_WORDS_MAX_CHARS + 1))

    def test_execute_tool_call_uses_the_allowlist(self) -> None:
        (
            success,
            success_tokens,
            success_memory_ids,
            success_pending_actions,
        ) = chatbot.execute_tool_call(
            types.SimpleNamespace(
                name="count_english_words",
                arguments=json.dumps({"text": "I study English every day."}),
            )
        )
        (
            rejected,
            rejected_tokens,
            rejected_memory_ids,
            rejected_pending_actions,
        ) = chatbot.execute_tool_call(
            types.SimpleNamespace(name="not_registered", arguments="{}")
        )

        self.assertEqual(json.loads(success)["result"]["word_count"], 5)
        self.assertTrue(json.loads(success)["ok"])
        self.assertEqual(success_tokens, 0)
        self.assertEqual(success_memory_ids, [])
        self.assertEqual(success_pending_actions, [])
        self.assertFalse(json.loads(rejected)["ok"])
        self.assertEqual(json.loads(rejected)["error"], "unknown tool")
        self.assertEqual(rejected_tokens, 0)
        self.assertEqual(rejected_memory_ids, [])
        self.assertEqual(rejected_pending_actions, [])

    def test_search_memory_returns_public_data_and_runtime_metadata(self) -> None:
        client = FakeClient()
        collection = FakeCollection()
        store = chatbot.LongTermMemoryStore("", "", collection)
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user often confuses check-in and check-out.",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0],
                )
            ],
            source="automatic",
        )[0]
        before = deepcopy(collection.records)

        result = chatbot.search_memory(
            client,
            store,
            "the user's travel English weakness",
        )

        public_memory = result.data["memories"][0]
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(public_memory["memory_id"], memory_id)
        self.assertEqual(public_memory["memory_type"], "semantic")
        self.assertEqual(public_memory["importance_score"], 4)
        self.assertEqual(public_memory["semantic_similarity"], 0.9)
        self.assertEqual(result.token_usage, 3)
        self.assertEqual(result.used_memory_ids, [memory_id])
        self.assertEqual(collection.records, before)

    def test_search_memory_validates_the_agent_query(self) -> None:
        client = FakeClient()
        store = fake_memory_store()

        with self.assertRaises(ValueError):
            chatbot.search_memory(client, store, 123)
        with self.assertRaises(ValueError):
            chatbot.search_memory(client, store, "   ")
        with self.assertRaises(ValueError):
            chatbot.search_memory(
                client,
                store,
                "q" * (chatbot.MEMORY_SEARCH_MAX_CHARS + 1),
            )

    def test_remember_memory_creates_a_policy_checked_pending_action(self) -> None:
        result = chatbot.remember_memory(
            content="  The user plans to take IELTS next May.  ",
            memory_type="semantic",
            reason="explicit_request",
        )

        self.assertEqual(result.data["status"], "accepted_for_commit")
        self.assertEqual(result.data["memory_type"], "semantic")
        self.assertEqual(result.token_usage, 0)
        self.assertEqual(result.used_memory_ids, [])
        self.assertEqual(
            result.pending_actions,
            [
                {
                    "type": "remember_memory",
                    "candidate": {
                        "content": "The user plans to take IELTS next May.",
                        "memory_type": "semantic",
                    },
                    "reason": "explicit_request",
                }
            ],
        )

    def test_remember_memory_rejects_sensitive_content_before_commit(self) -> None:
        result = chatbot.remember_memory(
            content="The user's API key is sk-example-secret-1234.",
            memory_type="semantic",
            reason="explicit_request",
        )

        self.assertEqual(result.data["status"], "rejected")
        self.assertIn("敏感", result.data["reason"])
        self.assertEqual(result.pending_actions, [])

    def test_request_forget_memory_only_creates_an_approval_request(self) -> None:
        store = fake_memory_store()
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user plans to take IELTS next May.",
                    memory_type="semantic",
                    importance_score=5,
                    embedding=[1.0],
                )
            ],
            source="agent",
        )[0]

        result = chatbot.request_forget_memory(
            store,
            memory_id=memory_id,
            reason="The user explicitly asked to remove it.",
        )

        self.assertEqual(result.data["status"], "pending_approval")
        self.assertEqual(result.data["memory_id"], memory_id)
        self.assertEqual(store.count(), 1)
        self.assertEqual(result.pending_actions[0]["type"], "forget_memory")

    def test_memory_preview_is_exact_and_bounded(self) -> None:
        store = fake_memory_store()
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="A memory that is longer than the preview.",
                    memory_type="episodic",
                    importance_score=2,
                    embedding=[1.0],
                )
            ],
            source="agent",
        )[0]

        self.assertEqual(store.get_preview(memory_id, max_chars=8), "A memory...")
        self.assertIsNone(store.get_preview("missing-id"))

    def test_model_can_answer_without_calling_a_tool(self) -> None:
        client = FakeClient()
        client.responses.create_results = [
            fake_model_response("I have finished my homework.", [], 9)
        ]

        result = chatbot.run_model_with_tools(
            client,
            [{"role": "user", "content": "Give me an example."}],
            fake_memory_store(),
        )

        self.assertEqual(result["reply"], "I have finished my homework.")
        self.assertEqual(result["response_tokens"], 9)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["tool_steps"], 0)
        self.assertEqual(result["used_memory_ids"], [])
        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(client.responses.calls[0]["tool_choice"], "auto")
        self.assertFalse(client.responses.calls[0]["parallel_tool_calls"])

    def test_tool_result_is_returned_with_the_matching_call_id(self) -> None:
        client = FakeClient()
        tool_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-123",
            name="count_english_words",
            arguments=json.dumps({"text": "I study English every day."}),
        )
        client.responses.create_results = [
            fake_model_response("", [tool_call], 11),
            fake_model_response("這句英文共有 5 個單字。", [], 17),
        ]
        input_messages = [
            {"role": "user", "content": "請精確計算 I study English every day."}
        ]

        result = chatbot.run_model_with_tools(
            client,
            input_messages,
            fake_memory_store(),
        )

        final_call = client.responses.calls[1]
        function_output = final_call["input"][-1]
        self.assertEqual(result["reply"], "這句英文共有 5 個單字。")
        self.assertEqual(result["response_tokens"], 28)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["tool_steps"], 1)
        self.assertEqual(result["used_memory_ids"], [])
        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertEqual(final_call["tool_choice"], "auto")
        self.assertEqual(final_call["input"][0], input_messages[0])
        self.assertEqual(final_call["input"][1].call_id, "call-123")
        self.assertEqual(function_output["call_id"], "call-123")
        self.assertEqual(json.loads(function_output["output"])["result"]["word_count"], 5)

    def test_agent_loop_can_execute_consecutive_tool_calls(self) -> None:
        client = FakeClient()
        first_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-a",
            name="count_english_words",
            arguments=json.dumps({"text": "I study English every day."}),
        )
        second_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-b",
            name="count_english_words",
            arguments=json.dumps(
                {"text": "I have studied English for three years."}
            ),
        )
        client.responses.create_results = [
            fake_model_response("", [first_call], 11),
            fake_model_response("", [second_call], 13),
            fake_model_response("B 比 A 多 2 個單字。", [], 17),
        ]

        result = chatbot.run_model_with_tools(
            client,
            [{"role": "user", "content": "請比較 A 和 B 的英文單字數量。"}],
            fake_memory_store(),
        )

        final_input = client.responses.calls[2]["input"]
        self.assertEqual(result["reply"], "B 比 A 多 2 個單字。")
        self.assertEqual(result["response_tokens"], 41)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["tool_steps"], 2)
        self.assertEqual(result["used_memory_ids"], [])
        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertTrue(
            all(
                call["tool_choice"] == "auto"
                for call in client.responses.calls
            )
        )
        self.assertEqual(final_input[2]["call_id"], "call-a")
        self.assertEqual(final_input[4]["call_id"], "call-b")

    def test_agent_can_search_memory_then_use_another_tool(self) -> None:
        client = FakeClient()
        store = fake_memory_store()
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user often confuses check-in and check-out.",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0],
                )
            ],
            source="automatic",
        )[0]
        search_call = types.SimpleNamespace(
            type="function_call",
            call_id="search-1",
            name="search_memory",
            arguments=json.dumps({"query": "travel English weakness"}),
        )
        count_call = types.SimpleNamespace(
            type="function_call",
            call_id="count-1",
            name="count_english_words",
            arguments=json.dumps(
                {"text": "The user often confuses check-in and check-out."}
            ),
        )
        client.responses.create_results = [
            fake_model_response("", [search_call], 11),
            fake_model_response("", [count_call], 13),
            fake_model_response("找到相關記憶，該句共有 9 個英文單字。", [], 17),
        ]

        result = chatbot.run_model_with_tools(
            client,
            [{"role": "user", "content": "找出我的旅遊英文弱點並計算單字。"}],
            store,
        )

        search_observation = json.loads(
            client.responses.calls[1]["input"][-1]["output"]
        )
        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertEqual(result["tool_steps"], 2)
        self.assertEqual(result["response_tokens"], 41)
        self.assertEqual(result["tool_tokens"], 3)
        self.assertEqual(result["used_memory_ids"], [memory_id])
        self.assertEqual(search_observation["result"]["count"], 1)
        self.assertNotIn("token_usage", search_observation["result"])
        self.assertNotIn("used_memory_ids", search_observation["result"])
        self.assertEqual(
            search_observation["result"]["memories"][0]["memory_id"],
            memory_id,
        )

    def test_agent_collects_pending_memory_actions_until_final_answer(self) -> None:
        client = FakeClient()
        remember_call = types.SimpleNamespace(
            type="function_call",
            call_id="remember-1",
            name="remember_memory",
            arguments=json.dumps(
                {
                    "content": "The user plans to take IELTS next May.",
                    "memory_type": "semantic",
                    "reason": "explicit_request",
                }
            ),
        )
        client.responses.create_results = [
            fake_model_response("", [remember_call], 11),
            fake_model_response("好，我會記住。", [], 17),
        ]

        result = chatbot.run_model_with_tools(
            client,
            [{"role": "user", "content": "請記住我明年五月要考 IELTS。"}],
            fake_memory_store(),
        )

        observation = json.loads(client.responses.calls[1]["input"][-1]["output"])
        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertEqual(result["tool_steps"], 1)
        self.assertEqual(len(result["pending_actions"]), 1)
        self.assertEqual(
            observation["result"]["status"],
            "accepted_for_commit",
        )
        self.assertNotIn("pending_actions", observation["result"])

    def test_agent_loop_stops_before_executing_a_tool_beyond_the_limit(self) -> None:
        client = FakeClient()
        first_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-1",
            name="count_english_words",
            arguments=json.dumps({"text": "First sentence."}),
        )
        blocked_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-2",
            name="count_english_words",
            arguments=json.dumps({"text": "Second sentence."}),
        )
        client.responses.create_results = [
            fake_model_response("", [first_call], 11),
            fake_model_response("", [blocked_call], 13),
        ]

        with (
            patch.object(chatbot, "MAX_AGENT_STEPS", 1),
            patch.object(
                chatbot,
                "execute_tool_call",
                wraps=chatbot.execute_tool_call,
            ) as execute_tool,
        ):
            result = chatbot.run_model_with_tools(
                client,
                [{"role": "user", "content": "Count both sentences."}],
                fake_memory_store(),
            )

        self.assertEqual(result["stop_reason"], "max_steps")
        self.assertEqual(result["tool_steps"], 1)
        self.assertEqual(result["response_tokens"], 24)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["used_memory_ids"], [])
        self.assertEqual(execute_tool.call_count, 1)
        self.assertEqual(len(client.responses.calls), 2)

    def test_agent_loop_reports_api_errors(self) -> None:
        client = FakeClient()

        with patch.object(
            client.responses,
            "create",
            side_effect=RuntimeError("offline"),
        ):
            result = chatbot.run_model_with_tools(
                client,
                [{"role": "user", "content": "Count these words."}],
                fake_memory_store(),
            )

        self.assertEqual(result["stop_reason"], "api_error")
        self.assertEqual(result["tool_steps"], 0)
        self.assertEqual(result["response_tokens"], 0)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["used_memory_ids"], [])

    def test_agent_loop_reports_unexpected_tool_errors(self) -> None:
        client = FakeClient()
        tool_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-error",
            name="count_english_words",
            arguments=json.dumps({"text": "Count me."}),
        )
        client.responses.create_results = [
            fake_model_response("", [tool_call], 11)
        ]

        with patch.object(
            chatbot,
            "execute_tool_call",
            side_effect=RuntimeError("unexpected failure"),
        ):
            result = chatbot.run_model_with_tools(
                client,
                [{"role": "user", "content": "Count these words."}],
                fake_memory_store(),
            )

        self.assertEqual(result["stop_reason"], "tool_error")
        self.assertEqual(result["tool_steps"], 1)
        self.assertEqual(result["response_tokens"], 11)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["used_memory_ids"], [])

    def test_agent_loop_rejects_an_empty_final_response(self) -> None:
        client = FakeClient()
        client.responses.create_results = [fake_model_response("   ", [], 7)]

        result = chatbot.run_model_with_tools(
            client,
            [{"role": "user", "content": "Answer me."}],
            fake_memory_store(),
        )

        self.assertEqual(result["stop_reason"], "empty_response")
        self.assertEqual(result["tool_steps"], 0)
        self.assertEqual(result["response_tokens"], 7)
        self.assertEqual(result["tool_tokens"], 0)
        self.assertEqual(result["used_memory_ids"], [])

    def test_context_budget_includes_reserved_tool_tokens(self) -> None:
        memory = chatbot.ShortTermMemory(client=FakeClient(), max_input_tokens=50)
        memory.add_user_message("current question")

        stats = memory.prepare_context(reserved_input_tokens=20)

        self.assertEqual(stats["input_tokens"], 10)
        self.assertEqual(stats["reserved_input_tokens"], 20)
        self.assertEqual(stats["estimated_input_tokens"], 30)

    def test_load_settings_reads_environment_overrides(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict(
                chatbot.os.environ,
                {
                    "MEMORA_MODEL": "test-model",
                    "MEMORA_EMBEDDING_MODEL": "test-embedding",
                    "MEMORA_DB_PATH": f"{temp_dir}/db",
                    "MEMORA_PROFILE_PATH": f"{temp_dir}/profile.json",
                    "MEMORA_MAX_AGENT_STEPS": "2",
                    "MEMORA_LOG_LEVEL": "debug",
                },
                clear=True,
            ),
        ):
            settings = chatbot.load_settings()

        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.embedding_model, "test-embedding")
        self.assertEqual(settings.memory_db_path, chatbot.Path(temp_dir) / "db")
        self.assertEqual(
            settings.user_profile_path,
            chatbot.Path(temp_dir) / "profile.json",
        )
        self.assertEqual(settings.max_agent_steps, 2)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_agent_step_setting_must_be_a_positive_integer(self) -> None:
        for invalid_value in ("0", "-1", "many"):
            with (
                self.subTest(value=invalid_value),
                patch.dict(
                    chatbot.os.environ,
                    {"MEMORA_MAX_AGENT_STEPS": invalid_value},
                    clear=True,
                ),
                self.assertRaises(ValueError),
            ):
                chatbot.load_settings()

    def test_startup_validation_requires_api_key(self) -> None:
        settings = chatbot.Settings(
            model="test-model",
            embedding_model="test-embedding",
            memory_db_path=chatbot.Path("unused-db"),
            user_profile_path=chatbot.Path("unused-profile.json"),
            max_agent_steps=1,
            log_level="INFO",
        )

        with (
            patch.dict(chatbot.os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"),
        ):
            chatbot.validate_startup(settings)

    def test_startup_validation_creates_data_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = chatbot.Settings(
                model="test-model",
                embedding_model="test-embedding",
                memory_db_path=chatbot.Path(temp_dir) / "database" / "memories",
                user_profile_path=chatbot.Path(temp_dir) / "profile" / "user.json",
                max_agent_steps=1,
                log_level="INFO",
            )
            with patch.dict(
                chatbot.os.environ,
                {"OPENAI_API_KEY": "test-key"},
                clear=True,
            ):
                chatbot.validate_startup(settings)

            self.assertTrue(settings.memory_db_path.is_dir())
            self.assertTrue(settings.user_profile_path.parent.is_dir())

    def test_command_router_handles_management_commands_only(self) -> None:
        client = FakeClient()
        output = StringIO()
        memory = chatbot.ShortTermMemory(client)
        store = fake_memory_store()
        profile_store = chatbot.UserProfileStore(chatbot.Path("missing-profile.json"))

        with contextlib.redirect_stdout(output):
            handled = chatbot.handle_command(
                "help",
                client,
                memory,
                store,
                profile_store,
            )
            natural_language = chatbot.handle_command(
                "Please explain the present perfect.",
                client,
                memory,
                store,
                profile_store,
            )

        self.assertTrue(handled)
        self.assertFalse(natural_language)
        self.assertIn("Commands:", output.getvalue())

    def test_forget_command_keeps_explicit_approval_boundary(self) -> None:
        client = FakeClient()
        memory = chatbot.ShortTermMemory(client)
        store = fake_memory_store()
        profile_store = chatbot.UserProfileStore(chatbot.Path("missing-profile.json"))
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user plans to take IELTS next May.",
                    memory_type="semantic",
                    importance_score=5,
                    embedding=[1.0],
                )
            ],
            source="agent",
        )[0]

        with (
            patch.object(builtins, "input", return_value="yes"),
            contextlib.redirect_stdout(StringIO()),
        ):
            handled = chatbot.handle_command(
                f"forget {memory_id}",
                client,
                memory,
                store,
                profile_store,
            )

        self.assertTrue(handled)
        self.assertEqual(store.count(), 0)

    def test_process_chat_turn_finishes_conversation_state(self) -> None:
        client = FakeClient()
        client.responses.create_results = [
            fake_model_response("I have finished my homework.", [], 9)
        ]
        memory = chatbot.ShortTermMemory(client)

        result = chatbot.process_chat_turn(
            "Give me an example.",
            client,
            memory,
            fake_memory_store(),
            chatbot.UserProfileStore(chatbot.Path("missing-profile.json")),
        )

        self.assertEqual(result["stop_reason"], "final_answer")
        self.assertEqual(
            memory.history,
            [
                {"role": "user", "content": "Give me an example."},
                {
                    "role": "assistant",
                    "content": "I have finished my homework.",
                },
            ],
        )
        self.assertEqual(memory.session_total_tokens, 9)

    def test_process_chat_turn_rolls_back_an_unfinished_request(self) -> None:
        client = FakeClient()
        memory = chatbot.ShortTermMemory(client)

        with (
            patch.object(
                chatbot,
                "run_model_with_tools",
                side_effect=RuntimeError("offline"),
            ),
            self.assertRaises(RuntimeError),
        ):
            chatbot.process_chat_turn(
                "This request will fail.",
                client,
                memory,
                fake_memory_store(),
                chatbot.UserProfileStore(chatbot.Path("missing-profile.json")),
            )

        self.assertEqual(memory.history, [])

    def test_agent_logs_actions_without_tool_content(self) -> None:
        client = FakeClient()
        private_text = "private sentence that must stay out of logs"
        tool_call = types.SimpleNamespace(
            type="function_call",
            call_id="call-private",
            name="count_english_words",
            arguments=json.dumps({"text": private_text}),
        )
        client.responses.create_results = [
            fake_model_response("", [tool_call], 5),
            fake_model_response("The sentence has eight words.", [], 7),
        ]

        with self.assertLogs(chatbot.logger, level="INFO") as captured:
            chatbot.run_model_with_tools(
                client,
                [{"role": "user", "content": "Count my sentence."}],
                fake_memory_store(),
            )

        log_output = "\n".join(captured.output)
        self.assertIn("action=count_english_words", log_output)
        self.assertNotIn(private_text, log_output)

    def test_main_recovers_from_one_failed_chat_turn(self) -> None:
        client = FakeClient()
        inputs = iter(["Please explain this.", "exit"])
        output = StringIO()

        with (
            patch.object(chatbot, "validate_startup"),
            patch.object(chatbot, "OpenAI", return_value=client),
            patch.object(
                chatbot,
                "create_long_term_memory_store",
                return_value=fake_memory_store(),
            ),
            patch.object(
                chatbot,
                "create_user_profile_store",
                return_value=chatbot.UserProfileStore(chatbot.Path("missing-profile.json")),
            ),
            patch.object(
                chatbot,
                "process_chat_turn",
                side_effect=RuntimeError("offline"),
            ),
            patch.object(builtins, "input", side_effect=lambda _="": next(inputs)),
            contextlib.redirect_stdout(output),
        ):
            chatbot.main()

        self.assertIn("這一輪暫時無法完成", output.getvalue())
        self.assertIn("Bye!", output.getvalue())

    def test_extracts_structured_memory_candidates(self) -> None:
        client = FakeClient()
        client.responses.extraction_result = chatbot.MemoryExtractionResult(
            should_remember=True,
            memories=[
                chatbot.MemoryCandidate(
                    content="使用者的英文程度是 B1。", memory_type="semantic"
                ),
                chatbot.MemoryCandidate(
                    content="使用者想加強旅遊英文。", memory_type="semantic"
                ),
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

    def test_embeds_candidates_in_api_index_order(self) -> None:
        client = FakeClient()
        embedded, tokens = chatbot.embed_memory_candidates(
            client,
            [
                chatbot.ScoredMemoryCandidate(
                    content="first", memory_type="semantic", importance_score=5
                ),
                chatbot.ScoredMemoryCandidate(
                    content="second", memory_type="episodic", importance_score=2
                ),
            ],
        )

        self.assertEqual([item.content for item in embedded], ["first", "second"])
        self.assertEqual(embedded[1].memory_type, "episodic")
        self.assertEqual(embedded[1].importance_score, 2)
        self.assertEqual(embedded[1].embedding, [1.0, 1.0])
        self.assertEqual(tokens, 6)

    def test_semantic_search_returns_highest_score_first(self) -> None:
        client = FakeClient()
        memories = [
            chatbot.EmbeddedMemoryCandidate(
                content="travel English",
                memory_type="semantic",
                importance_score=5,
                embedding=[1.0, 0.0],
            ),
            chatbot.EmbeddedMemoryCandidate(
                content="grammar book",
                memory_type="episodic",
                importance_score=2,
                embedding=[0.0, 1.0],
            ),
        ]
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        memory_ids = store.add(
            memories,
            source="automatic",
        )

        results, tokens = chatbot.semantic_search(client, "airport", store)

        self.assertEqual(results[0].content, "travel English")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].memory_id, memory_ids[0])
        self.assertEqual(results[0].source, "automatic")
        self.assertEqual(results[0].memory_type, "semantic")
        self.assertNotEqual(results[0].created_at, "unknown")
        self.assertEqual(tokens, 3)

    def test_long_term_store_lists_saved_memories(self) -> None:
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="使用者程度是 B1。",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0],
                )
            ],
            source="manual",
        )

        stored_memories = store.list_all()

        self.assertEqual(store.count(), 1)
        self.assertEqual(stored_memories[0].content, "使用者程度是 B1。")
        self.assertEqual(stored_memories[0].source, "manual")
        self.assertEqual(stored_memories[0].memory_type, "semantic")

    def test_retrieval_filters_low_scores_and_builds_background(self) -> None:
        client = FakeClient()
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="使用者程度是 B1。",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0],
                ),
                chatbot.EmbeddedMemoryCandidate(
                    content="曾問過現在完成式。",
                    memory_type="episodic",
                    importance_score=2,
                    embedding=[0.0],
                ),
            ],
            source="automatic",
        )

        memories, tokens = chatbot.retrieve_relevant_memories(
            client,
            store,
            "請安排適合我的課程",
        )
        messages = chatbot.build_background_messages(chatbot.UserProfile(), memories)

        self.assertEqual(len(memories), 1)
        self.assertEqual(tokens, 3)
        self.assertEqual(messages[0]["role"], "developer")
        self.assertIn("使用者程度是 B1。", messages[0]["content"])

    def test_background_messages_are_included_in_token_budget(self) -> None:
        memory = chatbot.ShortTermMemory(client=FakeClient())
        memory.add_user_message("請安排課程")
        background = [{"role": "developer", "content": "使用者程度是 B1。"}]

        stats = memory.prepare_context(background_messages=background)

        self.assertEqual(stats["context_messages"][0], background[0])
        self.assertEqual(stats["input_tokens"], 20)

    def test_user_profile_store_persists_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = chatbot.Path(temp_dir) / "user_profile.json"
            store = chatbot.UserProfileStore(path)
            store.set_english_level("b1")
            store.add_learning_goal("旅遊英文")
            store.add_preference("簡短例句")

            reloaded = chatbot.UserProfileStore(path).get()

        self.assertEqual(reloaded.english_level, "B1")
        self.assertEqual(reloaded.learning_goals, ["旅遊英文"])
        self.assertEqual(reloaded.preferences, ["簡短例句"])

    def test_background_combines_profile_and_retrieved_memories(self) -> None:
        profile = chatbot.UserProfile(
            english_level="B1",
            learning_goals=["旅遊英文"],
        )
        memory = chatbot.MemorySearchResult(
            memory_id="memory-1",
            content="昨天練習過機場報到。",
            memory_type="episodic",
            source="automatic",
            created_at="2026-09-10T00:00:00+00:00",
            distance=0.1,
            score=0.9,
        )

        messages = chatbot.build_background_messages(profile, [memory])

        self.assertIn("<user_profile>", messages[0]["content"])
        self.assertIn("English level: B1", messages[0]["content"])
        self.assertIn("<long_term_memories>", messages[0]["content"])
        self.assertIn("[episodic]", messages[0]["content"])

    def test_importance_scoring_preserves_candidate_order(self) -> None:
        client = FakeClient()
        client.responses.importance_result = chatbot.ImportanceAssessmentResult(
            assessments=[
                chatbot.ImportanceAssessment(candidate_index=1, importance_score=2),
                chatbot.ImportanceAssessment(candidate_index=0, importance_score=5),
            ]
        )

        scored, tokens = chatbot.score_memory_importance(
            client,
            [
                chatbot.MemoryCandidate(content="B2 exam", memory_type="semantic"),
                chatbot.MemoryCandidate(content="five words", memory_type="episodic"),
            ],
        )

        self.assertEqual([item.content for item in scored], ["B2 exam", "five words"])
        self.assertEqual([item.importance_score for item in scored], [5, 2])
        self.assertEqual(tokens, 10)

    def test_importance_scoring_rejects_missing_candidate(self) -> None:
        client = FakeClient()
        with self.assertRaises(ValueError):
            chatbot.score_memory_importance(
                client,
                [
                    chatbot.MemoryCandidate(content="first", memory_type="semantic"),
                    chatbot.MemoryCandidate(content="second", memory_type="semantic"),
                ],
            )

    def test_shared_storage_scores_embeds_and_persists_importance(self) -> None:
        client = FakeClient()
        client.responses.importance_result = chatbot.ImportanceAssessmentResult(
            assessments=[
                chatbot.ImportanceAssessment(candidate_index=0, importance_score=4)
            ]
        )
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())

        memory_ids, storage_tokens = chatbot.store_accepted_memories(
            client,
            store,
            [
                chatbot.MemoryCandidate(
                    content="The user studies business English.",
                    memory_type="semantic",
                )
            ],
            source="manual",
        )

        stored = store.list_all()[0]
        self.assertEqual(stored.memory_id, memory_ids[0])
        self.assertEqual(stored.importance_score, 4)
        self.assertEqual(storage_tokens, 13)

    def test_commit_remember_actions_revalidates_and_uses_write_pipeline(self) -> None:
        client = FakeClient()
        store = fake_memory_store()
        pending_actions = chatbot.remember_memory(
            content="The user plans to take IELTS next May.",
            memory_type="semantic",
            reason="explicit_request",
        ).pending_actions

        tokens = chatbot.commit_remember_actions(client, store, pending_actions)

        stored = store.list_all()
        self.assertEqual(tokens, 13)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].content, "The user plans to take IELTS next May.")
        self.assertEqual(stored[0].source, "agent")

    def test_commit_rejects_a_pending_action_modified_to_sensitive_data(self) -> None:
        client = FakeClient()
        store = fake_memory_store()
        pending_actions = chatbot.remember_memory(
            content="The user plans to take IELTS next May.",
            memory_type="semantic",
            reason="explicit_request",
        ).pending_actions
        pending_actions[0]["candidate"]["content"] = (
            "The user's API key is sk-example-secret-1234."
        )

        tokens = chatbot.commit_remember_actions(client, store, pending_actions)

        self.assertEqual(tokens, 0)
        self.assertEqual(store.count(), 0)

    def test_forget_review_requires_explicit_user_approval(self) -> None:
        store = fake_memory_store()
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user's goal is IELTS.",
                    memory_type="semantic",
                    importance_score=5,
                    embedding=[1.0],
                )
            ],
            source="agent",
        )[0]
        action = chatbot.request_forget_memory(
            store,
            memory_id,
            "The user asked to forget this goal.",
        ).pending_actions

        with (
            patch.object(builtins, "input", return_value="n"),
            contextlib.redirect_stdout(StringIO()),
        ):
            chatbot.review_forget_actions(store, action)
        self.assertEqual(store.count(), 1)

        with (
            patch.object(builtins, "input", return_value="yes"),
            contextlib.redirect_stdout(StringIO()),
        ):
            chatbot.review_forget_actions(store, action)
        self.assertEqual(store.count(), 0)

    def test_exact_duplicate_is_skipped_without_reconciliation_call(self) -> None:
        client = FakeClient()
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user prefers short examples.",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0, 0.0],
                )
            ],
            source="automatic",
        )
        incoming = chatbot.EmbeddedMemoryCandidate(
            content="  THE USER PREFERS SHORT   EXAMPLES.  ",
            memory_type="semantic",
            importance_score=4,
            embedding=[1.0, 0.0],
        )

        memory_ids, tokens = chatbot.reconcile_and_store_memory(
            client,
            store,
            incoming,
            source="manual",
        )

        reconciliation_calls = [
            call
            for call in client.responses.calls
            if call.get("text_format") is chatbot.MemoryReconciliationDecision
        ]
        self.assertEqual(memory_ids, [])
        self.assertEqual(tokens, 0)
        self.assertEqual(store.count(), 1)
        self.assertEqual(reconciliation_calls, [])

    def test_reconciliation_updates_mutable_state_in_place(self) -> None:
        client = FakeClient()
        collection = FakeCollection()
        store = chatbot.LongTermMemoryStore("", "", collection)
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user's current English level is B1.",
                    memory_type="semantic",
                    importance_score=4,
                    embedding=[1.0, 0.0],
                )
            ],
            source="automatic",
        )[0]
        before = store.list_all()[0]
        client.responses.reconciliation_result = (
            chatbot.MemoryReconciliationDecision(
                action="update",
                target_memory_id=memory_id,
                reason="The user provided a newer current level.",
            )
        )

        affected_ids, tokens = chatbot.reconcile_and_store_memory(
            client,
            store,
            chatbot.EmbeddedMemoryCandidate(
                content="The user's current English level is B2.",
                memory_type="semantic",
                importance_score=5,
                embedding=[2.0, 0.0],
            ),
            source="manual",
        )

        updated = store.list_all()[0]
        self.assertEqual(affected_ids, [memory_id])
        self.assertEqual(tokens, 10)
        self.assertEqual(store.count(), 1)
        self.assertEqual(updated.memory_id, memory_id)
        self.assertEqual(updated.content, "The user's current English level is B2.")
        self.assertEqual(updated.created_at, before.created_at)
        self.assertEqual(updated.last_accessed_at, before.last_accessed_at)
        self.assertEqual(updated.importance_score, 5)
        self.assertGreaterEqual(
            chatbot.parse_utc_timestamp(updated.updated_at),
            chatbot.parse_utc_timestamp(before.updated_at),
        )
        self.assertEqual(collection.records[0]["embedding"], [2.0, 0.0])

    def test_reconciliation_keeps_distinct_events(self) -> None:
        client = FakeClient()
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user completed airport English yesterday.",
                    memory_type="episodic",
                    importance_score=2,
                    embedding=[1.0, 0.0],
                )
            ],
            source="automatic",
        )
        client.responses.reconciliation_result = (
            chatbot.MemoryReconciliationDecision(
                action="keep_both",
                reason="These are distinct learning events.",
            )
        )

        memory_ids, tokens = chatbot.reconcile_and_store_memory(
            client,
            store,
            chatbot.EmbeddedMemoryCandidate(
                content="The user completed restaurant English today.",
                memory_type="episodic",
                importance_score=2,
                embedding=[1.0, 0.0],
            ),
            source="automatic",
        )

        self.assertEqual(len(memory_ids), 1)
        self.assertEqual(tokens, 10)
        self.assertEqual(store.count(), 2)

    def test_reconciliation_sends_uncertain_conflict_to_review(self) -> None:
        client = FakeClient()
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="The user's goal is IELTS.",
                    memory_type="semantic",
                    importance_score=5,
                    embedding=[1.0, 0.0],
                )
            ],
            source="automatic",
        )[0]
        client.responses.reconciliation_result = (
            chatbot.MemoryReconciliationDecision(
                action="review",
                target_memory_id=memory_id,
                reason="The possible change is not explicit enough.",
            )
        )

        memory_ids, tokens = chatbot.reconcile_and_store_memory(
            client,
            store,
            chatbot.EmbeddedMemoryCandidate(
                content="The user may no longer prepare for IELTS.",
                memory_type="semantic",
                importance_score=4,
                embedding=[1.0, 0.0],
            ),
            source="automatic",
        )

        self.assertEqual(memory_ids, [])
        self.assertEqual(tokens, 10)
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.list_all()[0].content, "The user's goal is IELTS.")

    def test_reconciliation_rejects_an_unknown_target(self) -> None:
        client = FakeClient()
        client.responses.reconciliation_result = (
            chatbot.MemoryReconciliationDecision(
                action="skip",
                target_memory_id="invented-id",
                reason="Duplicate.",
            )
        )
        candidate = chatbot.MemorySearchResult(
            memory_id="real-id",
            content="existing",
            memory_type="semantic",
            source="automatic",
            created_at="2026-09-14T00:00:00+00:00",
            updated_at="2026-09-14T00:00:00+00:00",
            last_accessed_at="2026-09-14T00:00:00+00:00",
            importance_score=3,
            distance=0.1,
            score=0.9,
        )

        with self.assertRaises(ValueError):
            chatbot.decide_memory_reconciliation(
                client,
                chatbot.EmbeddedMemoryCandidate(
                    content="incoming",
                    memory_type="semantic",
                    importance_score=3,
                    embedding=[1.0],
                ),
                [candidate],
            )

    def test_recency_score_uses_seven_day_half_life(self) -> None:
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)

        self.assertAlmostEqual(chatbot.calculate_recency_score(now.isoformat(), now), 1)
        self.assertAlmostEqual(
            chatbot.calculate_recency_score((now - timedelta(days=7)).isoformat(), now),
            0.5,
        )
        self.assertAlmostEqual(
            chatbot.calculate_recency_score((now - timedelta(days=14)).isoformat(), now),
            0.25,
        )
        self.assertEqual(chatbot.calculate_recency_score("unknown", now), 0.5)

    def test_memory_recency_uses_the_newer_update_time(self) -> None:
        now = datetime(2026, 9, 15, tzinfo=timezone.utc)
        score = chatbot.calculate_memory_recency_score(
            last_accessed_at=(now - timedelta(days=14)).isoformat(),
            updated_at=(now - timedelta(days=7)).isoformat(),
            now=now,
        )

        self.assertAlmostEqual(score, 0.5)

    def test_retrieval_score_combines_relevance_importance_and_recency(self) -> None:
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)
        result = chatbot.MemorySearchResult(
            memory_id="memory-1",
            content="B2 exam",
            memory_type="semantic",
            source="automatic",
            created_at=now.isoformat(),
            last_accessed_at=(now - timedelta(days=7)).isoformat(),
            importance_score=5,
            distance=0.2,
            score=0.8,
        )

        self.assertAlmostEqual(chatbot.calculate_retrieval_score(result, now), 0.81)

    def test_touch_updates_only_selected_memory_and_preserves_metadata(self) -> None:
        collection = FakeCollection()
        store = chatbot.LongTermMemoryStore("", "", collection)
        memory_ids = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="first",
                    memory_type="semantic",
                    importance_score=5,
                    embedding=[1.0],
                ),
                chatbot.EmbeddedMemoryCandidate(
                    content="second",
                    memory_type="episodic",
                    importance_score=2,
                    embedding=[0.0],
                ),
            ],
            source="automatic",
        )
        before = deepcopy(collection.records)

        store.touch([memory_ids[0], memory_ids[0]])

        self.assertEqual(collection.records[1], before[1])
        self.assertEqual(collection.records[0]["metadata"]["source"], "automatic")
        self.assertEqual(collection.records[0]["metadata"]["importance_score"], 5)
        self.assertIn("last_accessed_at", collection.records[0]["metadata"])

    def test_delete_forgets_only_existing_memory_id(self) -> None:
        store = chatbot.LongTermMemoryStore("", "", FakeCollection())
        memory_id = store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="forget me",
                    memory_type="episodic",
                    importance_score=1,
                    embedding=[1.0],
                )
            ],
            source="manual",
        )[0]

        self.assertTrue(store.delete(memory_id))
        self.assertFalse(store.delete(memory_id))
        self.assertEqual(store.count(), 0)

    def test_legacy_metadata_uses_compatible_defaults(self) -> None:
        collection = FakeCollection()
        collection.add(
            ids=["legacy"],
            documents=["old memory"],
            embeddings=[[1.0]],
            metadatas=[
                {"source": "automatic", "created_at": "2026-09-01T00:00:00+00:00"}
            ],
        )
        stored = chatbot.LongTermMemoryStore("", "", collection).list_all()[0]

        self.assertEqual(stored.importance_score, 3)
        self.assertEqual(stored.last_accessed_at, stored.created_at)
        self.assertEqual(chatbot.read_importance_score({"importance_score": True}), 3)

    def test_search_does_not_touch_memory(self) -> None:
        client = FakeClient()
        collection = FakeCollection()
        store = chatbot.LongTermMemoryStore("", "", collection)
        store.add(
            [
                chatbot.EmbeddedMemoryCandidate(
                    content="airport English",
                    memory_type="semantic",
                    importance_score=3,
                    embedding=[1.0],
                )
            ],
            source="automatic",
        )
        before = deepcopy(collection.records)

        chatbot.semantic_search(
            client,
            "airport",
            store,
        )

        self.assertEqual(collection.records, before)

    def test_policy_rejects_sensitive_content_for_every_source(self) -> None:
        candidate = chatbot.MemoryCandidate(
            content="The user's API key is sk-example-secret-1234.",
            memory_type="semantic",
        )

        automatic = chatbot.apply_memory_policy([candidate], source="automatic")
        manual = chatbot.apply_memory_policy([candidate], source="manual")

        self.assertFalse(automatic.decisions[0].should_store)
        self.assertFalse(manual.decisions[0].should_store)
        self.assertIn("敏感", manual.decisions[0].reason)

    def test_policy_respects_explicit_request_for_transient_memory(self) -> None:
        candidate = chatbot.MemoryCandidate(
            content="請明天提醒使用者複習單字。",
            memory_type="episodic",
        )

        automatic = chatbot.apply_memory_policy([candidate], source="automatic")
        manual = chatbot.apply_memory_policy([candidate], source="manual")

        self.assertEqual(automatic.approved_memories, [])
        self.assertEqual(manual.approved_memories, [candidate])
        self.assertIn("明確要求", manual.decisions[0].reason)

    def test_old_memory_without_type_is_unclassified(self) -> None:
        collection = FakeCollection()
        collection.add(
            ids=["legacy"],
            documents=["舊記憶"],
            embeddings=[[1.0]],
            metadatas=[{"source": "automatic", "created_at": "unknown"}],
        )
        store = chatbot.LongTermMemoryStore("", "", collection)

        self.assertEqual(store.list_all()[0].memory_type, "unclassified")


if __name__ == "__main__":
    unittest.main()
