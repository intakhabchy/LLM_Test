import ollama

# Store the conversation history
messages = []

while True:
    # Get input from the user
    user_input = input("You: ")

    # Exit the chatbot if the user types "exit"
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation history
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Send the conversation history to the local Qwen model
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=messages
    )

    # Extract Qwen's response
    answer = response["message"]["content"]

    # Add Qwen's response to the conversation history
    messages.append({
        "role": "assistant",
        "content": answer
    })

    # Display Qwen's response
    print("Qwen:", answer)