"""Memora v0.2: a prompted but deliberately stateless chatbot."""

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

    print("Memora v0.2")
    print("輸入 exit 可以結束對話。")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_input,
        )

        print("Memora:", response.output_text)


if __name__ == "__main__":
    main()
