"""Tests for Memora's in-session conversation history."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
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

        self.assertIn("Stored 1 memory.", output.getvalue())
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
