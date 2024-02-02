import torch
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer

LOGGING=logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Llama-2-7b model and tokenizer
#model_name = "meta-llama/Llama-2-7b-hf"
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "openlm-research/open_llama_3b_v2"
#model_name = "afmck/testing-llama-tiny"
#model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Input text
input_text = "What is the capital of Colombia?"

# Tokenize input
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# Access model layers
embeddings = model.get_input_embeddings()

#attention = model.get_decoder()
#ffn = attention.layers[0].mlp
#output_layer = model.lm_head

# Pass input through model layers
embedded_input = embeddings(input_ids)  # Get embedded representation

#attention_output = attention(embedded_input)  # Get attention output
#ffn_output = ffn(attention_output)  # Apply feedforward network
#logits = output_layer(ffn_output)  # Get final logits

# Generate text using top-k or top-p sampling
# https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
with torch.no_grad():
    generated_ids = model.generate(
        #input_ids = input_ids,
        inputs_embeds = embedded_input,
        num_return_sequences=1,
        min_length=1,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
        num_beams = 6, 
        length_penalty= 1.5,
        no_repeat_ngram_size=1,
    )

print("embedded_input: ",embedded_input)
print("generated_ids:",generated_ids)


generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("generated_text_2:",generated_text)

