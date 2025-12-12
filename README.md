# 🎓 ScholarSync - AI Thesis Consultant

**ScholarSync** is an intelligent "Dosen Pembimbing AI" designed to help final-year students (pejuang skripsi) analyze, summarize, and extract insights from their reference documents.

Built with a **Serverless First** architecture, it combines the power of **LLMs (OpenAI)**, **Vector Search (Astra DB)**, and **Cloud Storage (Vercel Blob)** to provide accurate, citation-backed answers.

---

## 🚀 Key Features

### 🧠 Smart Analysis
- **Context-Aware Chat**: The AI understands your specific documents and answers questions based *only* on them.
- **Strict Context Isolation**: Chat sessions are isolated. Discussing "Project A" will never retrieve documents from "Project B".
- **Smart Citations**: Every answer includes clickable citations (`[[file.pdf|page]]`) that verifying the source instantly.

### 📄 Document Management
- **PDF Viewer**: Split-screen viewer to read the exact page cited by the AI without leaving the chat.
- **Persistent Cloud Storage**: Documents are securely stored in **Vercel Blob**, ensuring they remain available even in serverless deployments.
- **Easy Organization**: Create "Proyek" (Sessions) to group documents by topic (e.g., "Bab 1", "Tinjauan Pustaka").

### 🎨 Premium UI/UX
- **Glassmorphism Design**: Modern, clean, and responsive interface using TailwindCSS.
- **Interactive Dashboard**: Track your thesis progress, see monthly activity, and manage documents.
- **Export to Markdown**: Download your chat history as a formatted draft for your thesis.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Backend** | Flask (Python) | Serverless-ready web framework |
| **Frontend** | Jinja2 + TailwindCSS | Responsive server-side rendering |
| **AI / LLM** | LangChain + OpenAI | Orchestration and Reasoning Agent |
| **Database** | DataStax Astra DB | Serverless Vector Database for RAG |
| **Storage** | Vercel Blob | Persistent object storage for PDFs |
| **Deployment** | Vercel | Serverless hosting platform |

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository_url>
cd ScholarSync
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Config
Create a `.env` file in the root directory:

```env
# AI & Database
OPENAI_API_KEY=sk-...
ASTRA_DB_API_ENDPOINT=https://<your-db-id>-<region>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...

# Storage (Vercel Blob)
BLOB_READ_WRITE_TOKEN=vercel_blob_...

# Security
SECRET_KEY=your_secret_key_here
```

### 4. Run Locally
```bash
python app.py
```
Visit `http://localhost:5000` to start.

---

## ☁️ Deployment (Vercel)

This project is optimized for Vercel.

1.  Push code to GitHub.
2.  Import project in Vercel Dashboard.
3.  Add the **Environment Variables** from step 3.
4.  **Important**: Create a Blob Store in Vercel Storage tab and link it to get the `BLOB_READ_WRITE_TOKEN`.
5.  Deploy! 🚀

---

## 📝 Usage Guide

1.  **Login/Register**: Create an account to save your progress.
2.  **Dashboard**: See your stats or click **"Mulai Bab Baru"** to create a session.
3.  **Upload**: Upload your PDF references (e.g., journals, books).
4.  **Chat**: Ask questions like:
    *   *"Apa research gap dari paper A?"*
    *   *"Buatkan kerangka teori dari dokumen ini."*
    *   *"Sebutkan batasan masalah yang ada."*
5.  **Verify**: Click on the citations (e.g., `📄 file.pdf P.5`) to open the PDF viewer.

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

**Happy Researching! 🎓**
