"""Memora v0.6: summarize older turns instead of discarding them."""

from openai import OpenAI


MODEL = "gpt-5-mini"
MAX_RECENT_TURNS = 3
MAX_INPUT_TOKENS = 1000

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


def count_input_tokens(client: OpenAI, messages: list[dict[str, str]]) -> int:
    result = client.responses.input_tokens.count(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=messages,
    )
    return result.input_tokens


def format_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role'].upper()}]\n{message['content']}"
        for message in messages
    )


def update_summary(
    client: OpenAI,
    current_summary: str,
    new_messages: list[dict[str, str]],
):
    summary_input = f"""
Existing Summary:
{current_summary or "(empty)"}

New Messages:
{format_messages(new_messages)}
""".strip()

    response = client.responses.create(
        model=MODEL,
        instructions=SUMMARY_INSTRUCTIONS,
        input=summary_input,
    )
    return response.output_text.strip(), response.usage


def create_context_messages(
    summary: str,
    recent_messages: list[dict[str, str]],
    current_user_message: dict[str, str],
) -> list[dict[str, str]]:
    context_messages: list[dict[str, str]] = []

    if summary:
        context_messages.append(
            {
                "role": "developer",
                "content": (
                    "The following is a compact summary of earlier conversation. "
                    "Use it as background context. If it conflicts with recent "
                    f"messages, prefer the recent messages.\n\n{summary}"
                ),
            }
        )

    return context_messages + recent_messages + [current_user_message]


def prepare_context(
    client: OpenAI,
    history: list[dict[str, str]],
    current_summary: str,
    summarized_count: int,
) -> dict[str, object]:
    current_user_message = history[-1]
    completed_messages = history[:-1]
    summary_token_usage = 0
    newly_summarized_count = 0
    target_summarized_count = max(
        0,
        len(completed_messages) - MAX_RECENT_TURNS * 2,
    )

    if target_summarized_count > summarized_count:
        new_messages = completed_messages[summarized_count:target_summarized_count]
        current_summary, usage = update_summary(
            client,
            current_summary,
            new_messages,
        )
        newly_summarized_count += len(new_messages)
        summary_token_usage += usage.total_tokens
        summarized_count = target_summarized_count

    recent_messages = completed_messages[summarized_count:]

    while True:
        context_messages = create_context_messages(
            current_summary,
            recent_messages,
            current_user_message,
        )
        input_tokens = count_input_tokens(client, context_messages)

        if input_tokens <= MAX_INPUT_TOKENS:
            return {
                "summary": current_summary,
                "summarized_count": summarized_count,
                "context_messages": context_messages,
                "input_tokens": input_tokens,
                "summary_token_usage": summary_token_usage,
                "newly_summarized_count": newly_summarized_count,
            }

        if len(recent_messages) < 2:
            raise ValueError("Summary 和目前的 User Message 已超過 Input Token Budget。")

        oldest_turn = recent_messages[:2]
        current_summary, usage = update_summary(
            client,
            current_summary,
            oldest_turn,
        )
        recent_messages = recent_messages[2:]
        summarized_count += 2
        newly_summarized_count += 2
        summary_token_usage += usage.total_tokens


def main() -> None:
    client = OpenAI()

    print("Memora v0.6")
    print("Commands: history, context, summary, exit")

    conversation_history: list[dict[str, str]] = []
    conversation_summary = ""
    summarized_message_count = 0
    last_context_messages: list[dict[str, str]] = []
    session_total_tokens = 0

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        if user_input.lower() == "history":
            print("\n--- Full Conversation History ---")

            if not conversation_history:
                print("（目前沒有對話紀錄）")
            else:
                for message in conversation_history:
                    print(f"{message['role']}: {message['content']}")

            print("----------------------------")
            continue

        if user_input.lower() == "summary":
            print("\n--- Conversation Summary ---")
            print(conversation_summary or "（目前還沒有產生 Summary）")
            print("----------------------------")
            continue

        if user_input.lower() == "context":
            print("\n--- Last Request Context ---")
            if not last_context_messages:
                print("（目前還沒有送出任何 Request）")
            else:
                for message in last_context_messages:
                    print(f"{message['role']}: {message['content']}")
            print("----------------------------")
            continue

        conversation_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            context_stats = prepare_context(
                client,
                conversation_history,
                conversation_summary,
                summarized_message_count,
            )
        except ValueError as error:
            conversation_history.pop()
            print("Error:", error)
            continue

        conversation_summary = context_stats["summary"]
        summarized_message_count = context_stats["summarized_count"]
        context_messages = context_stats["context_messages"]
        last_context_messages = [message.copy() for message in context_messages]
        session_total_tokens += context_stats["summary_token_usage"]

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=context_messages,
        )

        assistant_reply = response.output_text

        conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_reply,
            }
        )

        print("Memora:", assistant_reply)

        usage = response.usage
        session_total_tokens += usage.total_tokens

        print("\n--- Context Management ---")
        print("Stored history messages:", len(conversation_history))
        print("Summarized history messages:", summarized_message_count)
        print("Newly summarized messages:", context_stats["newly_summarized_count"])
        print("Context messages sent:", len(context_messages))
        print("Counted input tokens:", context_stats["input_tokens"])
        print("Actual input tokens:", usage.input_tokens)
        print("Output tokens:", usage.output_tokens)
        print("Summary update tokens:", context_stats["summary_token_usage"])
        print("Session total tokens:", session_total_tokens)
        print("--------------------------")


if __name__ == "__main__":
    main()
