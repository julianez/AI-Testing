import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
#model_name = "meta-llama/Llama-2-7b-chat-hf" 
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input embeddings (replace with your method for generating embeddings)
input_embeddings = torch.randn(1, 768)  # Example embeddings (batch size 1, sequence length 10, embedding dimension 768)
print(input_embeddings.shape)

# Input text
input_text = "Hello, how are you today?"

# Tokenize input
tokenized_input = tokenizer(input_text, return_tensors='pt')
input_ids = tokenized_input.input_ids

# Set model to eval mode
model.eval()

# Generate text from embeddings
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_length=1024,  # Increased for longer input embeddings
        num_beams=5,  # Adjust for beam search
    )

# Decode generated tokens
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
