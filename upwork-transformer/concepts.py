import torch
import logging

from transformers import LlamaTokenizer, LlamaForCausalLM, TextGenerationPipeline

# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the Llama-2-7b model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"

logging.info('LlamaTokenizer')
tokenizer = LlamaTokenizer.from_pretrained(model_name)
logging.info('LlamaForCausalLM')
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to(device)

logging.info('TextGenerationPipeline')
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Define the prompt
prompt = "Explain the concept of gravity in simple terms."

# Generate text with a system prompt to guide concept extraction
system_prompt = "[INST]\n<<SYS>>\nExplain the main concepts in the following text:\n<</SYS>>\n\n{user_prompt}".format(
    user_prompt=prompt
)

# Generate the response
response = pipeline(system_prompt, num_return_sequences=1)[0]["generated_text"]

# Print the extracted concepts
print("Extracted concepts:")
print(response)
