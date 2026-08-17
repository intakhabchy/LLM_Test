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
    n_results=1     # asks for only the 1 most similar result.
)

# Display the most similar document
print("Query:",query)
print("Result:",results["documents"][0][0])

# client initialized
# collection initialized
# documents created
# documents embedded in embeddings
# documents and embeddings added in collection with ids
# query created
# query embedded in query_embeddings
# results created with query() function running on collection and query_embedding as params
# Query and Result are printed
