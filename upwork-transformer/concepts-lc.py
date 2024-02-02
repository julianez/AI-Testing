import logging
import numpy as np
from tabulate import tabulate
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from datasets import load_dataset

# Configure the logging module
LOGGING=logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# GLOBAL PARAMETERS
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  
MAX_TOKENS = 256
N_GPU_LAYERS = 30 
N_BATCH = 512
TEMPERATURE = 0.1  # Sampling temperature

FS_ROWS = 3 # Number of few shot examples for concept
QUESTIONS = 1  # Number of questions
SEED = 34 # Seed for random questions

MODELS_PATH = "../../models"

MODEL_ID = "TheBloke/Llama-2-7B-GGUF"
MODEL_BASENAME = "llama-2-7b.Q4_K_M.gguf"

MODELS_TOKEN_PATH ="../../models/Llama-2-7b-hf"


# Define a function to generate concept using few-shot examples
def concept_question(llm, question, examples):
    
    prompt = fr'''{" ".join([f"Q: {example['question']} -> C: {example['title'].replace('_',' ')}" for example in examples])} Q: {question}'''
    print (prompt);
    return llm.create_completion(prompt,
                 stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
                 echo=False # Echo the prompt back in the output
                 )

def answer_baseline(llm, question):

    answer = llm(question,
                 echo=False, # Echo the prompt back in the output
                 max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE
                 )
    print(answer)
    answer_base = answer['choices'][0]['text']
    return answer_base

def answer_concept(llm, question, concept, examples):
    # Create the embedding
    
    prompt = f"Based on: {concept}, {question}"

    embedding = llm.create_embedding(prompt)
    
    answer = llm(prompt,
                 echo=False,
                 max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE                
                 )
    
    print(answer)
    answer_concept = answer['choices'][0]['text']

    return answer_concept

logging.info("Using Llamacpp for GGUF/GGML quantized models")
model_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_BASENAME,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
kwargs = {
            "model_path": model_path,
            "f16_kv":True,
            "embedding":True,
            "verbose":False,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
            "n_gpu_layers": N_GPU_LAYERS  # set this based on your GPU
}    
        
logging.info("Loading model")
llm = Llama(**kwargs)

logging.info("Loading Dataset")
dataset = load_dataset("squad_v2")

# Extract titles from the dataset
titles = dataset["train"]["title"]

# Get unique titles and their indices
unique_titles, indices = np.unique(titles, return_index=True)

# Select examples corresponding to unique titles
filtered_train = dataset["train"].select(indices)

# Filter the dataset
filtered_train = filtered_train.filter(lambda example: example["question"].find(example["title"].replace('_',' ')) != -1)

# Specify the number of random rows for few-shot
random_sample = filtered_train.shuffle(SEED).select(range(FS_ROWS))

#question
filtered_validation = dataset["validation"].filter(lambda example: example["question"].find(example["title"].replace('_',' ')) != -1)


filtered_validation = filtered_validation.shuffle().select(range(QUESTIONS))

# Initialize an empty list to store rows
concept_data = []

for validation_row in filtered_validation:
    # Get context, question, and expected concept
    context = validation_row["context"]
    question = validation_row["question"]
    expected_concept = validation_row["title"]
    answer_ds = validation_row["answers"]["text"]

    # Call concept_question function (Few shot )
    concept_fs = concept_question(llm, question, random_sample)
    
    answer_base = answer_baseline(llm, question)

    concept = concept_fs['choices'][0]['text'].split('C: ')[1].strip()

    answer_c = answer_concept(llm, question, concept, random_sample)

    concept_data.append((question, 
                         expected_concept.replace('_',' '), 
                         concept,
                         answer_base,
                         answer_c,
                         ))
 

# Print the output array with structured formatting
print(tabulate(concept_data,headers=['Question','Expected Concept','FS Concept','Answer Base', 'Answer Concept'],maxcolwidths=[40, None, None, 70, 70] ))


