from PyPDF2 import PdfReader
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain

reader = PdfReader('./SSRN_R1.pdf')

print("Reader:", reader) 

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

print("raw_text[:100]:", raw_text[:100]) 

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

print("len(texts):", len(texts)) 
print("texts[0]:", texts[0]) 

embeddings = LlamaCppEmbeddings(model_path="../../llama.cpp/models/7B/llama-2-7b.Q4_K_M.gguf")

docsearch = FAISS.from_texts(texts, embeddings)

print("docsearch:", docsearch) 

chain = load_qa_chain(LlamaCpp(), chain_type="stuff")

query = "who are the authors of the article?"

print("query:", query)

docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)