import ollama
import chromadb

# Create a local vector database
client = chromadb.PersistentClient(path="./rag_db")
collection = client.get_or_create_collection(name="knowledge")

# Read your own knowledge
with open("knowledge.txt","r") as file:
    text = file.read()

# Convert the knowledge into an embedding
embedding = ollama.embed(
    model="nomic-embed-text",
    input=text
)["embeddings"][0]

# Store the knowledge and its embedding
collection.add(
    ids=["1"],
    documents=[text],
    embeddings=[embedding]
)

# Ask a question
question = "When was the library established?"
# question = "When will the library open?"
# question = "How many books does the library have?"
# question = "লাইব্রেরী কবে খুলবে?"
# question = "লাইব্রেরীতে কয়টি বই আছে?"
# question = "বই সংখ্যা কত?"

# Convert the question into an embedding
question_embedding = ollama.embed(
    model="nomic-embed-text",
    input=question
)["embeddings"][0]

# Retrieve relevant information from the vector database
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)

context=results["documents"][0][0]

# Give the retrieved information to Qwen
response=ollama.chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role":"user",
            "content":f"Answer using only this information:\n\n{context}\n\nQuestion: {question}"
        }
    ]
)

# Display the answer
print("Qwen:",response["message"]["content"])