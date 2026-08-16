import ollama

messages = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model = "qwen2.5:1.5b",
        messages = messages
    )

    answer = response["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": answer
    })

    print("Qwen: ",answer)