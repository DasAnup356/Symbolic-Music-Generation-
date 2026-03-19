import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_tutor(index_path="rag/faiss_index", api_key=None):
    """
    Setup the RAG-based AI Tutor.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}. Run rag/indexer.py first.")

    # Initialize embeddings (must match indexer)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load vector store
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Initialize LLM
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found. Using retrieval-only mode (returning relevant chunks).")
        llm = None
    else:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, openai_api_key=api_key)

    # Define prompt template
    template = """
    You are Jules, an AI Tutor for the Symbolic Music Generation project.
    Your goal is to help the user understand the project's codebase, documentation, and the underlying deep learning concepts.
    Use the following context from the project to answer the user's question.
    If the context doesn't contain the answer, use your general knowledge but mention it's not from the project's specific docs.
    Be encouraging and technical yet clear.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    if llm:
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    else:
        # Retrieval-only mode
        return vector_store.as_retriever(search_kwargs={"k": 3})

def ask_tutor(query, tutor):
    """
    Query the tutor and return the answer.
    """
    if isinstance(tutor, RetrievalQA):
        # LLM-powered QA
        response = tutor.invoke(query)
        return response["result"]
    else:
        # Retrieval-only: return content of retrieved chunks
        docs = tutor.invoke(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return f"RETRIEVAL-ONLY MODE:\nHere are relevant excerpts from the project:\n\n{context}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Tutor for Symbolic Music Generation")
    parser.add_argument("query", type=str, help="The question you want to ask.")
    parser.add_argument("--api-key", type=str, help="OpenAI API Key (optional).")
    args = parser.parse_args()

    try:
        tutor = setup_tutor(api_key=args.api_key)
        answer = ask_tutor(args.query, tutor)
        print("\n" + "="*80)
        print("JULES - AI TUTOR")
        print("="*80)
        print(f"QUESTION: {args.query}")
        print("-" * 80)
        print(answer)
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
