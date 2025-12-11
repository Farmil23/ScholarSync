# 🎓 ScholarSync

**ScholarSync** is a professional **Retrieval-Augmented Generation (RAG)** system designed to intelligently query and retrieve information from documents.

Built with **LangChain**, **DataStax Astra DB**, and **Streamlit**, it allows users to have natural language conversations with their PDF libraries, featuring hybrid search and reranking for high-precision results.

## 🚀 Features

- **Hybrid Search**: Combines Scalar Vector Search with Keyword Search (simulated via Reranking in this version).
- **Advanced Reranking**: Uses `CrossEncoder` to refine search results for better relevance.
- **Conversational Memory**: Remembers context across the chat session.
- **Streamlit Interface**: Clean and responsive UI for easy interaction.
- **Astra DB Integration**: Scalable vector storage in the cloud.

## 🛠️ Tech Stack

- **Framework**: Python, Streamlit
- **LLM**: OpenAI (GPT-3.5/4) or HuggingFace (Local fallback)
- **Vector Database**: DataStax Astra DB
- **Orchestration**: LangChain
- **Embeddings**: OpenAI `text-embedding-3-small` / `all-MiniLM-L6-v2`

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd ScholarSync
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   # OpenAI (Optional, strictly recommended for quality)
   OPENAI_API_KEY=sk-...

   # Astra DB (Required)
   ASTRA_DB_API_ENDPOINT=https://<your-db-id>-<region>.apps.astra.datastax.com
   ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
   ```

## 📖 Usage

### 1. Ingest Documents
Place your PDF files into the `data/` folder, then run the ingestion script to process and upload them to Astra DB.

```bash
# Ensure you have PDFs in existing 'data' folder
python ingest.py
```

### 2. Run the Application
Launch the Streamlit interface:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `ingest.py`: Script to parse PDFs and load embeddings into Astra DB.
- `flask_app/`: Alternative Flask backend (if applicable).
- `data/`: Folder for source PDF documents.
- `requirements.txt`: Python package dependencies.
