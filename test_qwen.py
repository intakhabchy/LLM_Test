# This script downloads a new model for low configuration, the next procedure for training
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face model name
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download and load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")