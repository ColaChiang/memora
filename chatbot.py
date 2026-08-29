"""Memora v0.5: keep recent turns inside a bounded context window."""

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


def count_input_tokens(client: OpenAI, messages: list[dict[str, str]]) -> int:
    result = client.responses.input_tokens.count(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=messages,
    )
    return result.input_tokens


def build_context_window(
    client: OpenAI,
    history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Keep whole recent turns, then trim until the input fits the budget."""
    current_user_message = history[-1]
    completed_messages = history[:-1]
    recent_messages = completed_messages[-MAX_RECENT_TURNS * 2 :]
    context_messages = recent_messages + [current_user_message]

    while True:
        input_tokens = count_input_tokens(client, context_messages)

        if input_tokens <= MAX_INPUT_TOKENS:
            return context_messages, input_tokens

        if len(context_messages) == 1:
            raise ValueError("目前的 User Message 本身已超過 Input Token Budget。")

        context_messages = context_messages[2:]


def main() -> None:
    client = OpenAI()

    print("Memora v0.5")
    print("Commands: history, context, exit")

    conversation_history: list[dict[str, str]] = []
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
            context_messages, counted_input_tokens = build_context_window(
                client,
                conversation_history,
            )
        except ValueError as error:
            conversation_history.pop()
            print("Error:", error)
            continue

        last_context_messages = [message.copy() for message in context_messages]
        skipped_message_count = len(conversation_history) - len(context_messages)

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
        print("Context messages sent:", len(context_messages))
        print("Older messages skipped:", skipped_message_count)
        print("Counted input tokens:", counted_input_tokens)
        print("Actual input tokens:", usage.input_tokens)
        print("Output tokens:", usage.output_tokens)
        print("Session total tokens:", session_total_tokens)
        print("--------------------------")


if __name__ == "__main__":
    main()
