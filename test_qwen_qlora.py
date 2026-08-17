from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Hugging Face model name
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

# Confirm that 4-bit quantization is enabled
print(model.config.quantization_config)