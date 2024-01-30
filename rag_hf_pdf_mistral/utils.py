from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

MODELS_PATH = "../../models"

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  
N_GPU_LAYERS = 30 
N_BATCH = 512

# this is specific to Llama-2. 
system_prompt = """Eres un asistente útil y usarás el contexto proporcionado para responder las preguntas del usuario. 
Lee el contexto dado antes de responder las preguntas y piensa paso a paso.
Responde en español. 
Si no puedes responder una pregunta del usuario basándote en el contexto proporcionado, infórmale. No uses ninguna otra información para responder al usuario. """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):

    if promptTemplate_type=="llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST

            print("\n\n"+prompt_template+"\n\n")
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""
            
            prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
            print("\n\n"+prompt_template+"\n\n")
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            print("\n\n"+prompt_template+"\n\n")
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected. 
        if history:
            prompt_template = system_prompt + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = system_prompt + """
            
            Context: {context}
            User: {question}
            Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory, 

def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):

    """
    Load a GGUF/GGML quantized model using LlamaCpp.

    This function attempts to load a GGUF/GGML quantized model using the LlamaCpp library.
    If the model is of type GGML, and newer version of LLAMA-CPP is used which does not support GGML,
    it logs a message indicating that LLAMA-CPP has dropped support for GGML.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - LlamaCpp: An instance of the LlamaCpp model if successful, otherwise None.

    Notes:
    - The function uses the `hf_hub_download` function to download the model from the HuggingFace Hub.
    - The number of GPU layers is set based on the device type.
    """

    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "verbose":False,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS  # set this based on your GPU
        
        logging.info("Returning model")
        return LlamaCpp(**kwargs)
    except:
        if "ggml" in model_basename:
            logging.INFO("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None
