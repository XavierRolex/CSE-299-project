import chromadb
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


local_path = "bank.pdf"
loader = UnstructuredPDFLoader(file_path=local_path)
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
chunks = text_splitter.split_documents(data)

# Directory for the Chroma database
db_dir = "./chroma_db"

# Initialize Chroma database from the document chunks and embeddings
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag",
    persist_directory=db_dir
)

# Persist the Chroma vector store to disk
vector_db.persist()

print(f"Chroma vector database saved in {db_dir}")