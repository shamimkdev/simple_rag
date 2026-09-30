import os
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.logger import logger

load_dotenv()

DB_DIR = "./vector_db"
# Use gemini-1.5-flash or gemini-2.0-flash
LLM_MODEL = "gemini-3.8-flash"


def get_embeddings():
    # Runs 100% locally on CPU — zero API 404 errors or rate limits
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_or_update_vectorstore(documents: List[Document]) -> Chroma:
    """Splits documents into chunks and indexes them into local ChromaDB."""
    logger.info(f"Starting chunking for {len(documents)} document segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks for vector embedding.")
    
    logger.info(f"Embedding chunks into ChromaDB at {DB_DIR} using local HuggingFace embeddings...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=DB_DIR
        )
        logger.info("Successfully updated vector database.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to build/update vector database: {e}", exc_info=True)
        raise e


def load_existing_vectorstore() -> Chroma | None:
    """Loads vectorstore from disk if it exists."""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logger.info(f"Existing vectorstore found at {DB_DIR}. Loading...")
        try:
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=get_embeddings()
            )
            logger.info("Vectorstore loaded successfully.")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load existing vectorstore: {e}", exc_info=True)
            return None
            
    logger.info("No existing vectorstore found.")
    return None


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved context chunks for the prompt."""
    logger.info(f"Formatting {len(docs)} retrieved context chunks for LLM prompt.")
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        formatted.append(f"[Source: {os.path.basename(source)}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def query_rag_pipeline(question: str, vectorstore: Chroma) -> str:
    """Queries vectorstore and generates an answer via Gemini."""
    logger.info(f"Processing query: '{question}'")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Explicitly pass the API key from environment variables
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )
        
        prompt_template = """You are a helpful personal assistant. Answer the user's question using ONLY the provided context below.
If you do not know the answer based on the context, state that the information is not in your documents.

Context:
{context}

Question:
{question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(question)
        logger.info("Query successfully answered by Gemini model.")
        return response
    except Exception as e:
        logger.error(f"Error during RAG pipeline query execution: {e}", exc_info=True)
        return "An error occurred while generating the answer. Check logs/ for details."