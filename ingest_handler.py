
import os
import shutil
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Reuse existing functions from utils/ingest
from utils import get_llm

def get_embeddings():
    # Force OpenAI/BytePlus or other lightweight API embeddings for Vercel
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Fallback to None or raise error if no API key, preventing heavy local model load
        raise ValueError("OPENAI_API_KEY is required for Vercel deployment to avoid heavy local dependencies.")

def ingest_file(file_path, user_id, original_filename):
    """
    Ingests a single file into Astra DB with user_id metadata.
    """
    print(f"📄 Processing {original_filename} for User: {user_id}")
    
    # 1. Load - Manual pypdf implementation to avoid heavy dependency
    from pypdf import PdfReader
    from langchain_core.documents import Document

    documents = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(
                    page_content=text, 
                    metadata={"page": i + 1, "source": original_filename}
                ))
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return 0
    
    # 2. Add Metadata
    # We add user_id to ensure we can filter later.
    for doc in documents:
        doc.metadata["user_id"] = user_id
        doc.metadata["source"] = original_filename
        
    # 3. Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        return 0

    # 4. Insert into Astra
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    
    if not api_endpoint or not token:
        raise ValueError("Astra DB credentials missing.")
        
    vstore = AstraDBVectorStore(
        embedding=get_embeddings(),
        collection_name="scholarsync_vector_db",
        api_endpoint=api_endpoint,
        token=token,
    )
    
    ids = vstore.add_documents(chunks)
    print(f"✅ Inserted {len(ids)} chunks for {original_filename}")
    return len(ids)
