# ingest_pdf.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

def ingest_pdf(pdf_path, persist_directory="./chroma_db", collection_name="local-rag"):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Generate embeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    # Create and persist the Chroma vector store
    db = Chroma.from_documents(docs, embedding, persist_directory=persist_directory, collection_name=collection_name)
    db.persist()
    print(f"Ingested {len(docs)} chunks from {pdf_path} into {persist_directory}")

if __name__ == "__main__":
    ingest_pdf("bank.pdf")
