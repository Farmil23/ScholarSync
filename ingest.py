
import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load env
load_dotenv()

# Configuration
DATA_DIR = "./data" # Ensure this matches where your PDFs are
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_embeddings():
    """
    Returns the embedding model.
    Using OpenAI Embeddings as per notebook (or fallback to HF if you want local).
    """
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        print("⚠️ OPENAI_API_KEY not found. Using HuggingFace embeddings (might be slower locally).")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_documents():
    print(f"🚀 Starting Ingestion from {DATA_DIR}...")
    
    # 1. Load Documents
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in data directory.")
        return

    documents = []
    for file_path in pdf_files:
        print(f"📄 Loading {file_path}...")
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())
    
    print(f"✅ Loaded {len(documents)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(chunks)} chunks.")

    # 3. Initialize Vector Store
    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        print("❌ Astra DB credentials missing in .env")
        return

    embedding = get_embeddings()
    
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="scholarsync_vector_db",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )

    # 4. Insert (or Update)
    # Note: AstraDBVectorStore.add_documents usually handles insertion.
    # To avoid duplicates, you might want to delete collection first or manage IDs.
    # For this demo, we just append.
    print("💾 Inserting chunks into Astra DB (this may take a moment)...")
    inserted_ids = vstore.add_documents(chunks)
    print(f"✅ Successfully inserted {len(inserted_ids)} chunks into Astra DB!")

if __name__ == "__main__":
    ingest_documents()
