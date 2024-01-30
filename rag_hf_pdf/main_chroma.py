import os
import logging

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from utils import get_prompt_template, load_quantized_model_gguf_ggml

LOGGING=logging

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=False,
)

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.disable(logging.CRITICAL)  # Disables everything
logging.info('QA Started')

# Create embeddings
embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )

# load the vectorstore
db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

retriever = db.as_retriever()

logging.info('Loading Model')
huggingLLM = load_quantized_model_gguf_ggml(MODEL_ID, MODEL_BASENAME, "cuda", LOGGING)
# get the prompt template and memory if set by the user.
logging.info('Loading Prompt Template')
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)



while(True):
    qa = RetrievalQA.from_chain_type(
        llm=huggingLLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
                "prompt": prompt,
            },
    )
    user_input = input("\n\nHola, en que puedo ayudarte?\n")
    result = qa({"query": user_input})
    #print(len(result['source_documents']))
    answer = result["result"].strip()
    print(answer)