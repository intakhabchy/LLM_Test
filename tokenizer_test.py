from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

text = "I love machine learning."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))