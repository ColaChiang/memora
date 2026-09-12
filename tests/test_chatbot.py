"""Tests for Memora's in-session conversation history."""

from copy import deepcopy
from io import StringIO
import builtins
import contextlib
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

    def get(self, *, include) -> dict[str, object]:
        return {
            "ids": [record["id"] for record in self.records],
            "documents": [record["document"] for record in self.records],
            "metadatas": [record["metadata"] for record in self.records],
        }

    def query(self, *, query_embeddings, n_results, include) -> dict[str, object]:
        records = self.records[:n_results]
        distances = [0.1 + index * 0.5 for index, _ in enumerate(records)]
        return {
            "ids": [[record["id"] for record in records]],
            "documents": [[record["document"] for record in records]],
            "metadatas": [[record["metadata"] for record in records]],
            "distances": [distances],
        }


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
        inputs = iter(["remember semantic 我的英文程度是 B1。", "memories", "exit"])
        output = StringIO()

        with (
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

        self.assertIn("New Embedded Memories", output.getvalue())
        self.assertIn("1. 我的英文程度是 B1。", output.getvalue())
        self.assertIn("Source: manual", output.getvalue())

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
                chatbot.MemoryCandidate(content="first", memory_type="semantic"),
                chatbot.MemoryCandidate(content="second", memory_type="episodic"),
            ],
        )

        self.assertEqual([item.content for item in embedded], ["first", "second"])
        self.assertEqual(embedded[1].memory_type, "episodic")
        self.assertEqual(embedded[1].embedding, [1.0, 1.0])
        self.assertEqual(tokens, 6)

    def test_semantic_search_returns_highest_score_first(self) -> None:
        client = FakeClient()
        memories = [
            chatbot.EmbeddedMemoryCandidate(
                content="travel English",
                memory_type="semantic",
                embedding=[1.0, 0.0],
            ),
            chatbot.EmbeddedMemoryCandidate(
                content="grammar book",
                memory_type="episodic",
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
                    embedding=[1.0],
                ),
                chatbot.EmbeddedMemoryCandidate(
                    content="曾問過現在完成式。",
                    memory_type="episodic",
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
