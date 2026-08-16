import ollama
import numpy as np

# Three example sentences
text1 = "The cat is sleeping."
text2 = "The kitten is asleep."
text3 = "The computer is running."

# Convert the first sentence into an embedding vector
embedding1 = ollama.embed(
    model="nomic-embed-text",
    input=text1
)["embeddings"][0]

# Convert the second sentence into an embedding vector
embedding2 = ollama.embed(
    model="nomic-embed-text",
    input=text2
)["embeddings"][0]

# Convert the third sentence into an embedding vector
embedding3 = ollama.embed(
    model="nomic-embed-text",
    input=text3
)["embeddings"][0]

# Display the first five numbers of each vector
print("Vector 1:", embedding1[:5])
print("Vector 2:", embedding2[:5])
print("Vector 3:", embedding3[:5])

# Calculate cosine similarity
similarity_1_2 = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
similarity_1_3 = np.dot(embedding1, embedding3) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding3))

print("Similarity (1 - 2):",similarity_1_2)
print("Similarity (1 - 3):",similarity_1_3)