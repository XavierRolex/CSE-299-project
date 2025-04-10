from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

# Load and process the PDF once
loader = PyPDFLoader("docs/bank.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Initialize Chroma vectorstore
embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(split_docs, embedding=embedding, persist_directory="./chroma_db")
retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=Ollama(model="llama3"))

qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
)

def handle_user_input(user_input):
    try:
        start = time.time()
        result = qa.run(user_input)
        duration = round(time.time() - start, 2)
        return f"{result}  \n⏱️ Response time: {duration}s"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
