import chromadb
import ollama

# Create a local ChromaDB database
client = chromadb.PersistentClient(path="./vector_db")

# Create a collection to store our documents
collection = client.get_or_create_collection(name="my_documents")

# Documents to store
documents = [
    "The cat is sleeping.",
    "The dog is running.",
    "The computer is running."
]

# Create embeddings for the documents
embeddings=[]

for document in documents:
    embedding = ollama.embed(
        model="nomic-embed-text",
        input=document
    )["embeddings"][0]

    embeddings.append(embedding)

# Store documents and their embeddings
collection.add(
    ids=["1","2","3"],
    documents=documents,
    embeddings=embeddings
)

# Search the database
query = "Which animal is asleep?"

query_embedding=ollama.embed(
    model="nomic-embed-text",
    input=query
)["embeddings"][0]

# Run query() on collection and query_embedding
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

# Display the most similar document
print("Query:",query)
print("Result:",results["documents"][0][0])