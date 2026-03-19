import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_index(docs_dir=".", index_path="rag/faiss_index"):
    """
    Build a FAISS index from the project's markdown files and python code.
    """
    print(f"Indexing documents from {docs_dir}...")

    # Load Markdown files (documentation and study guide)
    md_loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    md_docs = md_loader.load()

    # Load Python files (codebase)
    py_loader = DirectoryLoader(docs_dir, glob="**/*.py", loader_cls=TextLoader)
    py_docs = py_loader.load()

    all_docs = md_docs + py_docs
    print(f"Loaded {len(all_docs)} documents.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Initialize embeddings
    print("Initializing embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vector_store.save_local(index_path)
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    build_index()
