"""Memora v0.1: a deliberately stateless command-line chatbot."""

from openai import OpenAI


MODEL = "gpt-5-mini"


def main() -> None:
    client = OpenAI()

    print("Memora v0.1")
    print("輸入 exit 可以結束對話。")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Bye!")
            break

        response = client.responses.create(
            model=MODEL,
            input=user_input,
        )

        print("AI:", response.output_text)


if __name__ == "__main__":
    main()
