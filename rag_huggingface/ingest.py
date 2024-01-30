import os
import concurrent.futures
import logging

from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import WebBaseLoader

def load_url(url):
    loader = WebBaseLoader([url], encoding="utf-8")
    return loader.load()[0]

# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Ingest Started')

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

logging.info('Reading URLs from file')

with open('URLs-Scotiabank-CL.txt', 'r') as f:
     lines = f.readlines()

urls = [line.strip() for line in lines]

logging.info('Getting URLs webDocs')

threads = []
documents = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(load_url, urls)
    webDocs = list(results)

logging.info('Creating Embeddings')
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )

logging.info('Generating DB')

db = Chroma.from_documents(
        webDocs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )