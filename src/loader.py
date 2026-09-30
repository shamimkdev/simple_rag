import os
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from src.logger import logger


def load_excel(file_path: Path) -> List[Document]:
    """Converts Excel sheets into text rows for indexing."""
    logger.info(f"Loading Excel file: {file_path.name}")
    documents = []
    
    try:
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name).dropna(how="all")
            
            for index, row in df.iterrows():
                row_content = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                metadata = {
                    "source": str(file_path),
                    "sheet": sheet_name,
                    "row": index + 1
                }
                documents.append(Document(page_content=row_content, metadata=metadata))
                
        logger.info(f"Successfully loaded {len(documents)} rows from Excel file: {file_path.name}")
    except Exception as e:
        logger.error(f"Failed to load Excel file {file_path.name}: {e}", exc_info=True)
        
    return documents


def load_single_file(file_path: Path) -> List[Document]:
    """Routes files to appropriate loaders based on extension."""
    ext = file_path.suffix.lower()
    logger.info(f"Attempting to load document: {file_path.name} ({ext})")
    
    try:
        if ext == ".txt":
            docs = TextLoader(str(file_path), encoding="utf-8").load()
        elif ext == ".pdf":
            docs = PyPDFLoader(str(file_path)).load()
        elif ext in [".doc", ".docx"]:
            docs = Docx2txtLoader(str(file_path)).load()
        elif ext in [".xlsx", ".xls"]:
            docs = load_excel(file_path)
        else:
            logger.warning(f"Unsupported file extension: {ext} for file {file_path.name}")
            return []
            
        logger.info(f"Loaded {len(docs)} page/document segments from {file_path.name}")
        return docs
    except Exception as e:
        logger.error(f"Error loading file {file_path.name}: {e}", exc_info=True)
        return []


def load_directory(data_dir: str) -> List[Document]:
    """Loads all supported documents from the data folder."""
    logger.info(f"Scanning directory for documents: {data_dir}")
    all_docs = []
    supported_extensions = {".txt", ".pdf", ".doc", ".docx", ".xlsx", ".xls"}
    
    path = Path(data_dir)
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            docs = load_single_file(file_path)
            all_docs.extend(docs)
            
    logger.info(f"Total document segments loaded from directory {data_dir}: {len(all_docs)}")
    return all_docs