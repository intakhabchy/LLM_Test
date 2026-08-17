import ollama
import chromadb

# Create a local vector database
client = chromadb.PersistentClient(path="./document_db")
collection = client.get_or_create_collection(name="library")

# Load the document
with open("library.txt","r") as file:
    document = file.read()

# Split the document into smaller pieces
# chunks = [
#     "My Library was established in 2020.",
#     "It contains more than 5,000 books.",
#     "The library has sections for Science, History, Literature, and Computer Science.",
#     "The library opens at 9 AM and closes at 8 PM.",
#     "Members can borrow up to three books at a time.",
#     "Books can be borrowed for 14 days."
# ]

# Split the document into chunks of approximately 100 words
words = document.split()

chunks = [
    " ".join(words[i:i + 100])
    for i in range(0, len(words), 100)
]

# Create embeddings for each chunk
embeddings = []

for chunk in chunks:
    embedding = ollama.embed(
        model = "nomic-embed-text",
        input = chunk
    )["embeddings"][0]

    embeddings.append(embedding)

# Store the chunks and embeddings
collection.add(
    ids = [str(i) for i in range(len(chunks))],
    documents = chunks,
    embeddings = embeddings
)

# Ask a question
question = input("Question: ")

# Create an embedding for the question
question_embedding = ollama.embed(
    model = "nomic-embed-text",
    input = question
)["embeddings"][0]

# Find the mose relevant document chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results = 2
)

# Get the retrieved information
context = "\n".join(results["documents"][0])

# Send the retrieved information to Qwen
response = ollama.chat(
    model = "qwen2.5:1.5b",
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer using only the following information:\n\n"
                f"{context}\n\n"
                f"Question: {question}"
            )
        }
    ]
)

# Display Qwen's answer
print("Qwen: ",response["message"]["content"])