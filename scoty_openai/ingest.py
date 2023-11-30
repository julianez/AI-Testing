import os

from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

with open('URLs-Scotiabank-CL.txt', 'r') as f:
     lines = f.readlines()

urls = [line.strip() for line in lines]

loader = WebBaseLoader(urls)
webDocs = loader.load()

embeddings = AzureOpenAIEmbeddings()

db = Chroma.from_documents(
        webDocs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )