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

MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

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
prompt, memory = get_prompt_template(promptTemplate_type="mistral", history=False)


questions = [
    "¿Cuáles son los documentos obligatorios que se pedirán por un liquidador según las características únicas del siniestro y usted?",
    "¿Qué deben hacer los siniestrados en caso de liquidación directa por la Compañía?",
    "¿Cuánto tiempo tiene el asegurado o beneficiario para oposirse a una liquidación directa por la Compañía?",
    "¿En qué plazo deben entregar los documentos necesarios un liquidador según las características únicas del siniestro y usted?",
    "¿Qué información debe ilustrar e informar el liquidador a los siniestrados sobre las gestiones que les corresponden realizar?",
    "¿Cuál es el plazo máximo para emitir un preinforme de liquidación sobre la cobertura del siniestro y los daños producidos por un liquidador según las características únicas del siniestro y usted?",
    "¿Qué tipo de documentos son clásicamente considerados como 'opcionales' para un siniestro?",
    "¿Por qué se pide que los documentos clásicamente considerados como 'opcionales' no sean solicitados si no se les requiere por su liquidador según las características únicas del siniestro y usted?",
    "¿En qué plazo deben entregar los elementos o documentos indicados como obligatorios que fueron robados junto con un automóvil si existen en el caso del Robo de Vehículo?",
    "¿Qué deben hacer los siniestrados si sus daños son de gran magnitud y solo su liquidador puede determinar la Pérdida Total de su automóvil según las características únicas del siniestro y usted?",
]


# Looping through the list of questions
for question in questions:
    qa = RetrievalQA.from_chain_type(
        llm=huggingLLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
                "prompt": prompt,
            },
    )
    #user_input = input("\n\nHola, en que puedo ayudarte?\n")
    print("\nPregunta:",question)
    user_input = question
    result = qa({"query": user_input})
    #print(len(result['source_documents']))
    answer = result["result"].strip()
    print(answer)