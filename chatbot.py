"""Memora v0.4: inspect how token usage grows with conversation history."""

from openai import OpenAI


MODEL = "gpt-5-mini"

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


def main() -> None:
    client = OpenAI()

    print("Memora v0.4")
    print("輸入 history 可以查看對話紀錄，輸入 exit 可以結束對話。")

    conversation_history: list[dict[str, str]] = []
    session_total_tokens = 0

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        if user_input.lower() == "history":
            print("\n--- Conversation History ---")

            if not conversation_history:
                print("（目前沒有對話紀錄）")
            else:
                for message in conversation_history:
                    print(f"{message['role']}: {message['content']}")

            print("----------------------------")
            continue

        conversation_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=conversation_history,
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

        print("\n--- Token Usage ---")
        print("Messages sent:", len(conversation_history) - 1)
        print("Input tokens:", usage.input_tokens)
        print("Output tokens:", usage.output_tokens)
        print("Total tokens:", usage.total_tokens)
        print("Session total tokens:", session_total_tokens)
        print("-------------------")


if __name__ == "__main__":
    main()
