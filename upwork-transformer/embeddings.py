import logging
import numpy as np
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import DistilBertTokenizerFast, DistilBertModel

def ask_question_with_embeddings(question_text, context_text):
    question_embedding = embedding_model.encode(question_text)
    context_embedding = embedding_model.encode(context_text)

    prompt_template = PromptTemplate(
        inputs=["question_embedding", "context_embedding"],
        template="Answer the question based on the provided context: {question_embedding} {context_embedding}",
        input_variables=["question_embedding", "context_embedding"]
    )

    response = llm(prompt=prompt_template.format(
        question_embedding=question_embedding.tolist(),
        context_embedding=context_embedding.tolist()
    ))

    return response

def ask_question_with_concatenated_embeddings(question_text, context_text):


    question_embedding = llm.embed(question_text)
    print("question_embedding:",question_embedding)
    context_embedding = llm.embed(context_text)
    print("context_embedding:",context_embedding)

    concatenated_embedding = np.concatenate((question_embedding, context_embedding))

    prompt_template = PromptTemplate(
        inputs=["concatenated_embedding"],
        template="Answer the question based on the provided information: {concatenated_embedding}",
        input_variables=["concatenated_embedding"]
    )

    print(prompt_template.format(concatenated_embedding=concatenated_embedding.tolist()))

    response = llm(prompt=prompt_template.format(concatenated_embedding=concatenated_embedding.tolist()))

    return response

# GLOBAL PARAMETERS

CONTEXT_WINDOW_SIZE = 32768
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  
MAX_TOKENS = 256
N_GPU_LAYERS = 33 
N_BATCH = 64
TEMPERATURE = 0.1  # Sampling temperature

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MODELS_PATH = "../../models"

MODEL_ID = "TheBloke/Llama-2-7B-GGUF"
MODEL_BASENAME = "llama-2-7b.Q4_K_M.gguf"

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
            "verbose":True,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
            "n_gpu_layers": N_GPU_LAYERS  # set this based on your GPU
}    
        
logging.info("Loading model")
llm = Llama(**kwargs)

embedding_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

question = "Capital of France?"
context = "Paris is the most populous city in France."

answer = ask_question_with_concatenated_embeddings(question, context)
print(answer)  # Output: "The capital of France is Paris."

