import torch
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM

# Initialize the Llama2 model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Define a function to encode text into embeddings
def encode_text(text):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return input_ids

# Define a function to query the Llama2 LLM using embeddings
def query_llm(embeddings):
    embeddings_float = embeddings.float()
    
    # Compute the mean
    return torch.mean(embeddings_float, dim=1)

# Define a function to generate text from embeddings using the model
def generate_text(embeddings):
    with torch.no_grad():
        output = model.generate(input_ids=embeddings, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
text = "What is the capital of France?"
embeddings = encode_text(text)
query_result = query_llm(embeddings)

# Generate text from the embeddings
generated_text = generate_text(embeddings)

print("Original text:", text)
print("Generated text:", generated_text)