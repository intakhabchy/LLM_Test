import ollama

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = ollama.chat(
        model = "qwen2.5:1.5b",
        messages = [
            {"role": "user", "content": user_input}
        ]
    )

    print("Qwen:", response["message"]["content"])