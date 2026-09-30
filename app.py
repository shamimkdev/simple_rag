import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.logger import logger
from src.loader import load_directory, load_single_file
from src.rag_engine import (
    build_or_update_vectorstore,
    load_existing_vectorstore,
    query_rag_pipeline
)

load_dotenv()

st.set_page_config(page_title="Personal RAG Assistant", layout="wide")
st.title("📄 Personal RAG Knowledge Base")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize Session State
if "vectorstore" not in st.session_state:
    logger.info("Initializing Streamlit app session state...")
    st.session_state.vectorstore = load_existing_vectorstore()

# Sidebar: File Management
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload files (.txt, .pdf, .docx, .xlsx)",
        type=["txt", "pdf", "doc", "docx", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    if st.button("Index Uploaded Documents") and uploaded_files:
        logger.info(f"User clicked 'Index Uploaded Documents' with {len(uploaded_files)} files.")
        with st.spinner("Processing files..."):
            new_docs = []
            for file in uploaded_files:
                save_path = DATA_DIR / file.name
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                
                logger.info(f"Saved uploaded file to disk: {save_path}")
                docs = load_single_file(save_path)
                new_docs.extend(docs)
            
            if new_docs:
                st.session_state.vectorstore = build_or_update_vectorstore(new_docs)
                st.success(f"Indexed {len(new_docs)} document segments!")
            else:
                st.warning("No valid content could be extracted.")

    st.markdown("---")
    if st.button("Re-Index Entire `./data` Folder"):
        logger.info("User clicked 'Re-Index Entire ./data Folder'")
        with st.spinner("Re-indexing directory..."):
            all_docs = load_directory(str(DATA_DIR))
            if all_docs:
                st.session_state.vectorstore = build_or_update_vectorstore(all_docs)
                st.success("Vector store fully refreshed!")
            else:
                st.warning("No files found in `./data` directory.")

# Main Chat Interface
st.subheader("Ask Questions About Your Documents")

if st.session_state.vectorstore is None:
    st.info("Upload documents on the left sidebar or place files into the `./data` directory to start.")
else:
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        logger.info(f"User submitted query in Streamlit UI: '{user_query}'")
        with st.spinner("Searching and generating answer..."):
            answer = query_rag_pipeline(user_query, st.session_state.vectorstore)
            st.markdown("### Answer")
            st.write(answer)