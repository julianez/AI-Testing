import torch
import logging
import numpy as np
import torch.nn.functional as F
from tabulate import tabulate
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig


# Define a function to generate concept using few-shot examples
def concept_question(causal_model, question, examples):
   
    prompt = fr'''{" ".join([f"Q: {' '.join(text.split()[:20])} -> C: {title}" for title, text in zip(examples['title'],examples['text'])])} \n Q: {question} -> C:'''
    logging.debug("FS prompt : %s",prompt)

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        generated_ids = causal_model.generate(
            input_ids = input_ids,
            num_return_sequences=1,
            min_length=1,
            max_length=1000,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
            num_beams = 6, 
            length_penalty= 1.5,
            no_repeat_ngram_size=1,
        )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    concept = output.rsplit("C: ", 1)[-1]
    
    logging.debug("Concept from FS: %s",concept)
    
    return concept

def answer_baseline(causal_model, question):

    input_ids = tokenizer(question, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        generated_ids = causal_model.generate(
            input_ids = input_ids,
            num_return_sequences=1,
            min_length=1,
            max_length=250,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
            num_beams = 6, 
            length_penalty= 1.5,
            no_repeat_ngram_size=1,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    logging.debug("Answer baseline: %s", generated_text)

    return generated_text

def getTokenIds(concept,question):
    # Tokenize inputs
    tokenizer.pad_token = tokenizer.eos_token

    concept_ids = tokenizer(concept, return_tensors='pt')['input_ids'].to(device)
    question_ids = tokenizer(question, return_tensors='pt')['input_ids'].to(device)

    return concept_ids, question_ids

def padEmbeddings(embedding1,embedding2):
    logging.debug("Shape (Before) of embedding1_cpu: %s", embedding1.shape)
    logging.debug("Shape (Before) of embedding2_cpu: %s", embedding2.shape)

    if embedding2.size(1) > embedding1.size(1):
        # Calculate the difference in size along the second dimension
        diff = embedding2.size(1) - embedding1.size(1)
        # Create the padding tuple
        padding = (0, 0, 0, diff)
        # Pad the smaller tensor
        embedding1 = F.pad(embedding1, padding)
    else:
        # Calculate the difference in size along the second dimension
        diff = embedding1.size(1) - embedding2.size(1)
        # Create the padding tuple
        padding = (0, 0, 0, diff)
        # Pad the smaller tensor
        embedding2 = F.pad(embedding2, padding)


    logging.debug("Shape (After) of embedding1_cpu: %s", embedding1.shape)
    logging.debug("Shape (After) of embedding2_cpu: %s", embedding2.shape)

    return embedding1,embedding2

def combineEmbeddingMethod(embedding1,embedding2,weight1,weight2):
    # 2. Element-wise addition:
    combined_embedding = embedding1*weight1 + embedding2*weight2

    return combined_embedding

def answer_concept_combined(causal_model, question, concept):
    concept_ids, question_ids = getTokenIds(concept,question)

    # Access model layers
    embeddings = causal_model.get_input_embeddings()
    embeddings = embeddings.to(device)

    # Get embedded representation of the ids
    embedded_concept = embeddings(concept_ids)
    embedded_question = embeddings(question_ids)

    embedded_concept,embedded_question = padEmbeddings(embedded_concept,embedded_question)

    combined = combineEmbeddingMethod(embedded_concept,embedded_question,3,1)

    logging.debug("Combined Type: %s", type(combined))
    logging.debug("Combined Embedding: %s ",combined)

    with torch.no_grad():
        generated_ids = causal_model.generate(
            inputs_embeds = combined,
            num_return_sequences=1,
            min_length=1,
            max_length=250,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
            num_beams = 6, 
            length_penalty= 1.5,
            no_repeat_ngram_size=1,
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    logging.debug("Generated Text:",generated_text)
    return generated_text

# Configure the logging module
LOGGING=logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.debug("Device: %s", device)

FS_ROWS = 10 # Number of few shot examples for concept
QUESTIONS = 4  # Number of questions
SEED_1 = 34 # Seed for random questions
SEED_2 = 54 # Seed for random questions

# Load the Llama-2-7b model and tokenizer
#causal_model_path = "meta-llama/Llama-2-7b-hf"
#causal_model_path = 'openlm-research/open_llama_7b_v2'
causal_model_path = "openlm-research/open_llama_3b_v2"
#causal_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#causal_model_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

logging.info("Loading model for Causal")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(causal_model_path,legacy=False)
causal_model = LlamaForCausalLM.from_pretrained(causal_model_path,
                                         quantization_config=bnb_config,
                                         torch_dtype=torch.float16, 
                                         device_map='auto')

logging.info("Loading Dataset")
dataset = load_dataset('wikimedia/wikipedia', '20231101.en')

# few-shot
shuffled_dataset = dataset['train'].shuffle(seed=SEED_1)
random_sample = shuffled_dataset[:FS_ROWS]

#questions
shuffled_dataset = dataset['train'].shuffle(seed=SEED_2)
questions_dataset = shuffled_dataset[:QUESTIONS]

# Initialize an empty list to store rows
concept_data = []

for question in questions_dataset['text']:
        
        question = ' '.join(question.split()[:20])

        concept_fs = concept_question(causal_model, question, random_sample)
        
        answer_base = answer_baseline(causal_model, question)

        answer_concept = answer_concept_combined(causal_model, question, concept_fs)

        concept_data.append((question,
                            concept_fs,
                            answer_base,
                            answer_concept,
                            ))


# Print the output array with structured formatting
print(tabulate(concept_data,headers=['Question','FS Concept','Answer Base', 'Answer Concept'],maxcolwidths=[30, 20, 60, 60],stralign="left" ))


