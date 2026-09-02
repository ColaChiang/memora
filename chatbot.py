"""Memora v0.9: automatically extract long-term memory candidates."""

from openai import OpenAI


MODEL = "gpt-5-mini"
MAX_INPUT_TOKENS = 1000
MAX_RECENT_TURNS = 3

SYSTEM_PROMPT = """
You are Memora, a personal English learning assistant.

Your goal is to help the user learn English clearly and efficiently.

Guidelines:
- Explain concepts in simple language.
- Keep answers focused on the user's question.
- Use short examples when helpful.
- Avoid unnecessary technical grammar terminology.
- Do not add unrelated motivational text.
""".strip()

SUMMARY_INSTRUCTIONS = """
You maintain a compact summary of an ongoing conversation.

Requirements:
- Merge the existing summary with the new messages.
- Preserve user facts, preferences, goals, corrections, decisions, and unresolved questions.
- Remove greetings, repetition, and unimportant wording.
- Do not invent information.
- If newer information corrects older information, keep the newer information.
- Keep the summary under 150 words.
- Use the same language as the conversation when possible.
- Return only the updated summary.
""".strip()

MEMORY_EXTRACTION_INSTRUCTIONS = """
You extract potential long-term memories from a user message
for a personal English learning assistant.

Extract only information that may be useful in future conversations, such as:
- The user's English level
- Long-term learning goals
- Stable learning preferences
- Recurring learning difficulties
- Relevant learning constraints

Do not extract:
- Greetings or casual conversation
- One-time requests
- Questions that only depend on the current context
- Information created by the assistant
- Uncertain assumptions
- Passwords, secret codes, or unnecessary sensitive data

Rewrite each memory as a short, self-contained statement.
Return exactly NONE, or one or more lines in this format:
MEMORY: <memory>
Do not add explanations or Markdown.
""".strip()


def extract_memory_candidates(client: OpenAI, user_input: str) -> tuple[list[str], int, str]:
    response = client.responses.create(
        model=MODEL,
        instructions=MEMORY_EXTRACTION_INSTRUCTIONS,
        input=user_input,
    )
    raw_output = response.output_text.strip()
    candidates: list[str] = []

    if raw_output.upper() != "NONE":
        for line in raw_output.splitlines():
            if line.strip().upper().startswith("MEMORY:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    candidates.append(candidate)

    return candidates, response.usage.total_tokens, raw_output


class ShortTermMemory:
    def __init__(
        self,
        client: OpenAI,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        summary_instructions: str = SUMMARY_INSTRUCTIONS,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_recent_turns: int = MAX_RECENT_TURNS,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.summary_instructions = summary_instructions
        self.max_input_tokens = max_input_tokens
        self.max_recent_turns = max_recent_turns

        self.history: list[dict[str, str]] = []
        self.summary = ""
        self.summarized_message_count = 0
        self.last_context: list[dict[str, str]] = []
        self.session_total_tokens = 0

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def rollback_last_user_message(self) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

    @staticmethod
    def format_messages(messages: list[dict[str, str]]) -> str:
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        count = self.client.responses.input_tokens.count(
            model=self.model,
            instructions=self.system_prompt,
            input=messages,
        )
        return count.input_tokens

    def update_summary(self, new_messages: list[dict[str, str]]) -> int:
        summary_input = f"""
Existing summary:
{self.summary or "(empty)"}

New messages:
{self.format_messages(new_messages)}
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=self.summary_instructions,
            input=summary_input,
        )
        self.summary = response.output_text.strip()
        summary_tokens = response.usage.total_tokens
        self.session_total_tokens += summary_tokens
        return summary_tokens

    def create_context_messages(
        self,
        recent_messages: list[dict[str, str]],
        current_user_message: dict[str, str],
    ) -> list[dict[str, str]]:
        context_messages: list[dict[str, str]] = []
        if self.summary:
            context_messages.append(
                {
                    "role": "developer",
                    "content": (
                        "The following is a compact summary of earlier conversation. "
                        "Use it as background context. If it conflicts with recent "
                        f"messages, prefer the recent messages.\n\n{self.summary}"
                    ),
                }
            )
        return context_messages + recent_messages + [current_user_message]

    def prepare_context(self) -> dict[str, object]:
        if not self.history:
            raise ValueError("Conversation History 是空的。")
        if self.history[-1]["role"] != "user":
            raise ValueError("建立 Context 前，最後一則訊息必須是 User Message。")

        completed_messages = self.history[:-1]
        current_user_message = self.history[-1]
        summary_token_usage = 0
        newly_summarized_count = 0
        recent_message_limit = self.max_recent_turns * 2
        target_summarized_count = max(
            0,
            len(completed_messages) - recent_message_limit,
        )

        if target_summarized_count > self.summarized_message_count:
            new_messages = completed_messages[
                self.summarized_message_count : target_summarized_count
            ]
            summary_token_usage += self.update_summary(new_messages)
            newly_summarized_count += len(new_messages)
            self.summarized_message_count = target_summarized_count

        recent_messages = completed_messages[self.summarized_message_count :]

        while True:
            context_messages = self.create_context_messages(
                recent_messages,
                current_user_message,
            )
            input_tokens = self.count_input_tokens(context_messages)

            if input_tokens <= self.max_input_tokens:
                return {
                    "context_messages": context_messages,
                    "input_tokens": input_tokens,
                    "summary_token_usage": summary_token_usage,
                    "newly_summarized_count": newly_summarized_count,
                }

            if len(recent_messages) < 2:
                raise ValueError(
                    "目前的 User Message 與摘要已超過 MAX_INPUT_TOKENS。"
                )

            oldest_turn = recent_messages[:2]
            summary_token_usage += self.update_summary(oldest_turn)
            newly_summarized_count += 2
            self.summarized_message_count += 2
            recent_messages = recent_messages[2:]

    def finish_turn(
        self,
        assistant_reply: str,
        context_messages: list[dict[str, str]],
        response_tokens: int,
    ) -> None:
        self.history.append({"role": "assistant", "content": assistant_reply})
        self.last_context = [message.copy() for message in context_messages]
        self.session_total_tokens += response_tokens

    def add_token_usage(self, total_tokens: int) -> None:
        self.session_total_tokens += total_tokens

    def get_status(self) -> dict[str, object]:
        return {
            "History messages": len(self.history),
            "Summarized messages": self.summarized_message_count,
            "Last context messages": len(self.last_context),
            "Summary available": bool(self.summary),
            "Session total tokens": self.session_total_tokens,
        }


def print_messages(title: str, messages: list[dict[str, str]]) -> None:
    print(f"\n--- {title} ---")
    if not messages:
        print("(empty)")
    else:
        for message in messages:
            print(f"{message['role']}: {message['content']}")
    print("----------------------------")


def main() -> None:
    client = OpenAI()
    memory = ShortTermMemory(client=client)
    memory_candidates: list[str] = []
    last_extraction_output = ""

    print("Memora v0.9")
    print(
        "Commands: history, context, summary, status, "
        "remember <text>, memories, extraction, exit"
    )

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        command = user_input.lower()
        should_auto_extract = True

        if command == "exit":
            print("Bye!")
            break
        if command == "history":
            print_messages("Full Conversation History", memory.history)
            continue
        if command == "context":
            print_messages("Last Request Context", memory.last_context)
            continue
        if command == "summary":
            print("\n--- Conversation Summary ---")
            print(memory.summary or "(empty)")
            print("----------------------------")
            continue
        if command == "memories":
            print("\n--- Memory Candidates ---")
            if not memory_candidates:
                print("(empty)")
            else:
                for index, candidate in enumerate(memory_candidates, start=1):
                    print(f"{index}. {candidate}")
            print("-------------------------")
            continue
        if command == "extraction":
            print("\n--- Last Extraction Output ---")
            print(last_extraction_output or "(not run)")
            print("------------------------------")
            continue
        if command == "status":
            print("\n--- Short-term Memory Status ---")
            for name, value in memory.get_status().items():
                print(f"{name}: {value}")
            print("Memory candidates:", len(memory_candidates))
            print("--------------------------------")
            continue
        if command == "remember":
            print("Usage: remember <text>")
            continue
        if command.startswith("remember "):
            candidate = user_input[len("remember ") :].strip()
            if not candidate:
                print("Usage: remember <text>")
                continue
            memory_candidates.append(candidate)
            print("Saved as Memory Candidate:", candidate)
            user_input = candidate
            should_auto_extract = False
            last_extraction_output = "(manual memory)"

        memory.add_user_message(user_input)
        try:
            memory_stats = memory.prepare_context()
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=memory_stats["context_messages"],
            )
        except Exception as error:
            memory.rollback_last_user_message()
            print("Request failed:", error)
            continue

        assistant_reply = response.output_text
        memory.finish_turn(
            assistant_reply=assistant_reply,
            context_messages=memory_stats["context_messages"],
            response_tokens=response.usage.total_tokens,
        )

        extracted_candidates: list[str] = []
        extraction_tokens = 0
        if should_auto_extract:
            try:
                (
                    extracted_candidates,
                    extraction_tokens,
                    last_extraction_output,
                ) = extract_memory_candidates(client, user_input)
                memory_candidates.extend(extracted_candidates)
                memory.add_token_usage(extraction_tokens)
            except Exception as error:
                last_extraction_output = f"Extraction failed: {error}"

        print("Memora:", assistant_reply)
        if extracted_candidates:
            print("\n--- New Memory Candidates ---")
            for candidate in extracted_candidates:
                print("-", candidate)
            print("-----------------------------")
        print("\n--- Memory Status ---")
        print("Newly summarized messages:", memory_stats["newly_summarized_count"])
        print("Context messages sent:", len(memory_stats["context_messages"]))
        print("Counted input tokens:", memory_stats["input_tokens"])
        print("Actual input tokens:", response.usage.input_tokens)
        print("Output tokens:", response.usage.output_tokens)
        print("Summary update tokens:", memory_stats["summary_token_usage"])
        print("Memory extraction tokens:", extraction_tokens)
        print("Memory candidates:", len(memory_candidates))
        print("Session total tokens:", memory.session_total_tokens)
        print("---------------------")


if __name__ == "__main__":
    main()
