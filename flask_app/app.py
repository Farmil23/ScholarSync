
import os
import uuid
import csv
import io
import json
from datetime import datetime
import queue
import threading
from langchain_core.callbacks import BaseCallbackHandler
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, Response, stream_with_context
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

# LangChain Imports (Core Only)
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, BaseTool
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Third-party (Lightweight)
from ddgs import DDGS
from gtts import gTTS
import io

# Local Imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import get_llm # Moved inside function due to import scope or removed
from ingest_handler import ingest_file
from flask_app.models import db, User, ChatSession, ChatMessage, Document, ActivityLog

app = Flask(__name__, instance_path='/tmp', instance_relative_config=True)
# Config
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Database Config for Vercel (Neon/Postgres)
database_url = os.getenv("DATABASE_URL")
if not database_url:
    database_url = os.getenv("CONNECTION_STRING")
is_vercel = os.environ.get('VERCEL') == '1' or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')

if not database_url:
    # CRITICAL: In serverless (Vercel), we CANNOT use SQLite/Memory for persistence.
    # We must enforce Neon/Postgres.
    print("❌ ERROR: DATABASE_URL is missing in environment variables.")
    raise RuntimeError("DATABASE_URL is missing. Please add it in Vercel Settings.")

# Clean up common copy-paste errors (e.g. "psql 'postgresql://...'")
if "psql" in database_url:
    import re
    # Extract just the URL part: postgres://...
    match = re.search(r"(postgres(?:ql)?://\S+)", database_url)
    if match:
        database_url = match.group(1).rstrip("'")
        print(f"⚠️ Cleaned DATABASE_URL: Detected 'psql' command, extracting URL.")

if "sqlite" in database_url and is_vercel:
    print("❌ ERROR: SQLite URL detected. Vercel does not support SQLite.")
    raise RuntimeError("SQLite is not supported. Please use the Neon PostgreSQL URL.")

if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
}
if is_vercel:
    app.config['UPLOAD_FOLDER'] = '/tmp'
else:
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'docs')

# Init Extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Localization Config ---
translations = {
    'en': {
        'dashboard_title': 'Dashboard',
        'welcome': 'Hello, Thesis Warrior! 👋',
        'subtitle': 'Focus on progress. What is your target today?',
        'new_chapter': 'Start New Chapter',
        'total_refs': 'Total References',
        'keep_growing': 'Keep growing!',
        'consultations': 'Consultations',
        'active_sessions': 'Active sessions',
        'thesis_progress': 'Thesis Progress',
        'target_week': 'Target This Week',
        'target_desc': 'Finish Chapter 2 Revision (Theory) before Friday meeting.',
        'doc_status': 'Document Status',
        'doc_ready': 'All references indexed and ready for AI analysis.',
        'daily_tip': 'Daily Tip',
        'tip_content': '"A good thesis is a finished thesis. Don\'t be a perfectionist on the first draft!"',
        'active_projects': 'Active Projects',
        'created': 'Created',
        'continue': 'Continue',
        'no_projects': 'No projects yet. Start a new consultation!',
        'doc_archive': 'Document Archive',
        'upload': 'Upload',
        'no_docs': 'No documents yet.',
        'upload_now': 'Upload Now',
        'search_placeholder': 'Search projects or documents...',
        'sort_newest': 'Newest',
        'sort_oldest': 'Oldest',
        'sort_name': 'Name (A-Z)',
        'modal_new_project': 'Start New Project',
        'project_name_label': 'Project / Chapter Name',
        'project_name_placeholder': 'Example: Chapter 2 Literature Review',
        'select_refs': 'Select Reference Documents',
        'ai_note': 'AI will only answer based on selected documents.',
        'cancel': 'Cancel',
        'create': 'Create Project',
        'upload_ref': 'Upload Reference',
        'select_pdf': 'Select PDF File',
        'drag_drop': 'Click or drag file here',
        'pdf_only': 'PDF files only (Max 10MB)',
        'processing': 'Processing...',
        'reading_doc': 'Reading and indexing document content...',
        'rename_doc': 'Rename Document',
        'new_name': 'New Name',
        'save': 'Save',
        'delete_doc': 'Delete Document?',
        'delete_confirm_desc': 'This document will be permanently deleted and cannot be restored.',
        'delete': 'Delete',
        'labels': ['Title', 'Chap 1', 'Chap 2', 'Chap 3', 'Defense', 'Done']
    },
    'id': {
        'dashboard_title': 'Dashboard',
        'welcome': 'Halo, Pejuang Skripsi! 👋',
        'subtitle': 'Fokus pada progres. Apa targetmu hari ini?',
        'new_chapter': 'Mulai Bab Baru',
        'total_refs': 'Total Referensi',
        'keep_growing': 'Terus bertambah!',
        'consultations': 'Konsultasi',
        'active_sessions': 'Sesi aktif',
        'thesis_progress': 'Progress Skripsi',
        'target_week': 'Target Minggu Ini',
        'target_desc': 'Selesaikan Revisi Bab 2 (Teori) sebelum bimbingan Jumat.',
        'doc_status': 'Status Dokumen',
        'doc_ready': 'Semua referensi telah terindeks dan siap dianalisis AI.',
        'daily_tip': 'Tips Hari Ini',
        'tip_content': '"Skripsi yang baik adalah skripsi yang selesai. Jangan perfeksionis di draft pertama!"',
        'active_projects': 'Proyek Aktif',
        'created': 'Dibuat',
        'continue': 'Lanjut',
        'no_projects': 'Belum ada proyek. Mulai konsultasi baru!',
        'doc_archive': 'Arsip Dokumen',
        'upload': 'Upload',
        'no_docs': 'Belum ada dokumen.',
        'upload_now': 'Upload Sekarang',
        'search_placeholder': 'Cari proyek atau dokumen...',
        'sort_newest': 'Terbaru',
        'sort_oldest': 'Terlama',
        'sort_name': 'Nama (A-Z)',
        'modal_new_project': 'Mulai Proyek Baru',
        'project_name_label': 'Nama Proyek / Bab',
        'project_name_placeholder': 'Contoh: Bab 2 Tinjauan Pustaka',
        'select_refs': 'Pilih Dokumen Referensi',
        'ai_note': 'AI hanya akan menjawab berdasarkan dokumen yang dipilih.',
        'cancel': 'Batal',
        'create': 'Buat Proyek',
        'upload_ref': 'Upload Referensi',
        'select_pdf': 'Pilih File PDF',
        'drag_drop': 'Klik atau drag file ke sini',
        'pdf_only': 'Hanya file PDF (Max 10MB)',
        'processing': 'Memproses...',
        'reading_doc': 'Sedang membaca dan mengindeks isi dokumen...',
        'rename_doc': 'Ganti Nama Dokumen',
        'new_name': 'Nama Baru',
        'save': 'Simpan',
        'delete_doc': 'Hapus Dokumen?',
        'delete_confirm_desc': 'Dokumen ini akan dihapus permanen dan tidak bisa dikembalikan.',
        'delete': 'Hapus',
        'labels': ['Judul', 'Bab 1', 'Bab 2', 'Bab 3', 'Sidang', 'Selesai']
    }
}

@app.context_processor
def inject_conf_var():
    lang = request.args.get('lang', request.cookies.get('lang', 'id'))
    if lang not in translations:
        lang = 'id'
    
    def t(key):
        return translations[lang].get(key, key)
    
    return dict(t=t, current_lang=lang, translations=translations)


# Setup Uploads
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except OSError:
    # Fallback to tmp if creation fails (e.g. read-only)
    app.config['UPLOAD_FOLDER'] = '/tmp'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Debug Handler (Temporary) ---
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    # Pass through HTTP errors
    if isinstance(e, ValueError): # Example check
        pass
    
    # Return JSON/Text traceback for debugging 500s in Vercel
    trace = traceback.format_exc()
    return f"<h1>Internal Server Error (Debug Mode)</h1><pre>{trace}</pre>", 500

# --- Database Setup (Run once) ---
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created/verified.")
        
        # Simple Auto-Migration for 'thesis_stage'
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('user')]
        if 'thesis_stage' not in columns:
            print("Migrating: Adding thesis_stage column to user table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE "user" ADD COLUMN thesis_stage INTEGER DEFAULT 0'))
                conn.commit()
            print("Migration successful.")

        # Simple Auto-Migration for 'file_url' in 'document'
        doc_columns = [c['name'] for c in inspector.get_columns('document')]
        if 'file_url' not in doc_columns:
            print("Migrating: Adding file_url column to document table...")
            with db.engine.connect() as conn:
                # Add column, set default null
                conn.execute(text('ALTER TABLE "document" ADD COLUMN file_url VARCHAR(512)'))
                conn.commit()
            print("Migration successful: file_url added.")
            
    except Exception as e:
        print(f"WARNING: Database connection failed during startup: {e}")
        # We generally do not want to stop the app here, 
        # so that the / route can still load and logs can be seen.

# --- RAG / Agent Logic ---

def get_astradb_retriever(user_id):
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    
    if os.getenv("OPENAI_API_KEY"):
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        raise ValueError("OPENAI_API_KEY required for Vercel.")

    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="scholarsync_vector_db",
        api_endpoint=api_endpoint,
        token=token,
    )
    
    # Filter by user_id string for Astra
    return vstore.as_retriever(
        search_kwargs={
            "k": 5, 
            "filter": {"user_id": str(user_id)}
        }
    )

def get_agent_executor(user_id, doc_names=None):
    """
    Creates an Agent Executor with access to the user's vector store.
    Includes Prompt Engineering for Smart Citations.
    """
    from langchain_core.prompts import PromptTemplate
    from utils import get_llm # Lazy import

    llm = get_llm()
    
    # 1. Retriever Tool
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    
    # Retrieve embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="scholarsync_vector_db",
        api_endpoint=api_endpoint,
        token=token,
    )
    
    # Build Filter
    # Base filter: Must match user_id
    filter_dict = {'user_id': str(user_id)}
    
    # Session Context Filter: If doc_names (specific context) is provided, restrict retrieval
    # This prevents the AI from citing "Group 1" doc in "Group 2" session.
    if doc_names:
        # Astra DB / LangChain filter syntax for "IN"
        if len(doc_names) == 1:
             filter_dict['source'] = doc_names[0]
        else:
             filter_dict['source'] = {'$in': doc_names}
    
    # Use the specific filter
    retriever = vstore.as_retriever(
        search_kwargs={
            'filter': filter_dict, 
            'k': 4 # Increased slightly for better context
        }
    )
    
    # --- Custom Lightweight Tools ---
    
    # 1. Retriever Tool (Manual wrapper)
    def retrieve_fn(query: str):
        # We need to manually execute retrieval
        # .invoke() or .get_relevant_documents() depending on version
        docs = retriever.get_relevant_documents(query)
        # Format for the LLM
        return "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 1)})\nContent: {d.page_content}" for d in docs])

    retriever_tool = Tool(
        name="search_user_documents",
        func=retrieve_fn,
        description="Searches and returns excerpts from the user's uploaded PDF documents. Input should be a specific search query."
    )
    
    # 2. Search Tool (Using ddgs directly)
    def search_fn(query: str):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No results found."
                return "\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])
        except Exception as e:
            return f"Search Error: {str(e)}"

    search_tool = Tool(
        name="web_search",
        func=search_fn, 
        description="Search the web for general knowledge, current events, or info not in the documents."
    )
    
    tools = [retriever_tool, search_tool]
    tool_map = {t.name: t for t in tools}
    
    # Format valid filenames for the prompt
    valid_files_str = ", ".join(doc_names) if doc_names else "No specific files (General Knowledge Mode)."

    # 3. Simple ReAct Prompt
    template = f"""You are Dr. Sync, an expert Academic Thesis Consultant.

Role & Behavior:
1. **Critical & Academic**: Don't just answer. Critique the student's question if it's vague. Suggest better academic phrasing.
2. **Evidence-Based**: ALWAYS use the `search_user_documents` tool first.
3. **Structured**: When asked about "Research Gap" or "Framework", provide a structured list.
4. **Language**: Use formal Indonesian (Bahasa Baku) mixed with standard English academic terms.

MANDATORY CITATION FORMAT:
When you use knowledge from a document, you MUST cite it using the EXACT filename found in the 'source' metadata field of the retrieved context.
Format: [[filename.pdf|page_number]]
VALID FILENAMES: [{valid_files_str}]

TOOLS:
------
You have access to the following tools:

{{tools}}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to the user, or if you do not need to use a tool, you must use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{{chat_history}}

Question: {{input}}
Thought:{{agent_scratchpad}}"""

    # 4. Custom Lightweight Agent Executor
    class SimpleReactExecutor:
        def __init__(self, llm, tools, prompt_template):
            self.llm = llm
            self.tools = {t.name: t for t in tools}
            self.prompt_template = prompt_template
            self.max_steps = 5
            
        def invoke(self, inputs, config=None):
            # Extract inputs
            user_input = inputs.get("input", "")
            chat_history = inputs.get("chat_history", "")
            callbacks = config.get("callbacks", []) if config else []
            
            # Helper to stream thought
            def log_token(text):
                for cb in callbacks:
                    if hasattr(cb, 'on_llm_new_token'):
                        cb.on_llm_new_token(text)
            
            # Prepare Prompt
            tools_desc = "\\n".join([f"{t.name}: {t.description}" for t in self.tools.values()])
            tool_names = ", ".join(self.tools.keys())
            
            scratchpad = ""
            
            import re
            
            for step in range(self.max_steps):
                # Format full prompt
                full_prompt = self.prompt_template.replace("{{tools}}", tools_desc) \
                                          .replace("{{tool_names}}", tool_names) \
                                          .replace("{{chat_history}}", chat_history) \
                                          .replace("{{input}}", user_input) \
                                          .replace("{{agent_scratchpad}}", scratchpad)
                
                # Call LLM
                # We assume LLM supports invoke() or .predict()
                # For BytePlus/OpenAI wrapper, we might use invoke() text in text out
                # If Custom LLM, it might stream?
                # We will just use invoke and simulate stream via callback manually if needed, 
                # OR if LLM streams, we capture it.
                # Since we replaced the built-in agent, we lose automatic streaming from LLM->Callback->Queue.
                # We need to manually pipe LLM output to callback if possible.
                # However, for simplicity/stability, let's just run blocking call first 
                # AND fake stream the thought process to the UI.
                
                response_text = self.llm.invoke(full_prompt)
                
                # Send to callback (Simulation)
                log_token(response_text)
                
                # Parse
                # Look for Action: ... Action Input: ...
                action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", response_text, re.DOTALL)
                
                if "Final Answer:" in response_text:
                    # Done
                    final_ans = response_text.split("Final Answer:")[-1].strip()
                    return {"output": final_ans}
                
                if action_match:
                    action_name = action_match.group(1).strip()
                    action_input = action_match.group(2).strip()
                    
                    # Tool Execution
                    if action_name in self.tools:
                        log_token(f"\n[Executing {action_name}...]\n")
                        try:
                             # Handle input like "query" or 'query'
                             action_input = action_input.strip('"').strip("'")
                             observation = self.tools[action_name].func(action_input)
                        except Exception as e:
                             observation = f"Error: {str(e)}"
                    else:
                        observation = f"Error: Tool {action_name} not found."
                        
                    scratchpad += f"{response_text}\nObservation: {observation}\n"
                    log_token(f"Observation: {observation}\n")
                else:
                    # No action found, maybe just thoughts or malformed.
                    # If it ends without Final Answer, we might just assume it's talking.
                    # Or force loop.
                    # For safety, break if no action and no final answer.
                    if "Do I need to use a tool? No" in response_text:
                         # It forgot Final Answer label?
                         return {"output": response_text}
                    return {"output": response_text}
                    
            return {"output": "Agent stopped (max steps reached)."}

    # Return our custom executor
    return SimpleReactExecutor(llm, tools, template)

# --- Logger ---
def log_activity(user_id, action, details=None):
    try:
        log = ActivityLog(user_id=user_id, action=action, details=details)
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        print(f"Failed to log activity: {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('landing.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            log_activity(user.id, 'login')
            
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
                
            return redirect(url_for('dashboard')) # Redirect to Dashboard
        else:
            flash('Login Failed. Check email and password.', 'danger')
            
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user specific stats
    docs_count = Document.query.filter_by(user_id=current_user.id).count()
    chats_count = ChatSession.query.filter_by(user_id=current_user.id).count()
    
    # Recent documents
    recent_docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.uploaded_at.desc()).limit(5).all()
    all_docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.uploaded_at.desc()).all()
    # Fetch ALL sessions for searchability/scrollability
    recent_sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    
    stats = {
        'docs_count': docs_count,
        'chats_count': chats_count
    }
    
    return render_template('dashboard.html', user=current_user, stats=stats, recent_docs=recent_docs, all_docs=all_docs, recent_sessions=recent_sessions)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar', 'danger')
            return redirect(url_for('register'))
            
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan, pilih yang lain', 'danger')
            return redirect(url_for('register'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Auto-Admin for specific email
        is_admin = (email == 'admin@scholarsync.com')
        
        new_user = User(username=username, email=email, password=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        log_activity(new_user.id, 'register')
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/chat')
@login_required
def chat_page():
    # Fetch all sessions for sidebar
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    return render_template('chat.html', user=current_user, sessions=sessions)

@app.route('/session/<int:session_id>/rename', methods=['PUT'])
@login_required
def rename_session(session_id):
    data = request.json
    new_title = data.get('title')
    
    if not new_title:
        return jsonify({'error': 'Title required'}), 400
        
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    session.title = new_title
    db.session.commit()
    
    return jsonify({'message': 'Session renamed', 'title': new_title})

@app.route('/documents', methods=['GET'])
@login_required
def get_documents():
    docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.uploaded_at.desc()).all()
    return jsonify([{
        'id': d.id,
        'filename': d.filename,
        'chunk_count': d.chunk_count,
        'uploaded_at': d.uploaded_at.isoformat()
    } for d in docs])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        user_id = str(current_user.id)
        # 1. Save Locally (Temporary for Ingestion)
        filename_secure = f"{user_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
        file.save(filepath)
        
        try:
            # 2. Ingest to Vector DB
            count = ingest_file(filepath, user_id, file.filename)
            if count == 0:
                 return jsonify({'error': 'Could not extract text from PDF'}), 400
            
            file_url = None
            
            # 3. Upload to Vercel Blob (if token exists)
            # Use basic requests or wrapper if available. Python SDK is unofficial.
            # Fallback: We won't implement complex Blob logic without SDK.
            # BUT, we can try to use vercel_blob if installed.
            try:
                import vercel_blob
                # Read file back
                with open(filepath, 'rb') as f:
                    # Upload with explicit Content-Type ensures browser knows to display likely
                    blob_resp = vercel_blob.put(filename_secure, f.read(), options={
                        'access': 'public',
                        'content_type': 'application/pdf' 
                    })
                    file_url = blob_resp.get('url')
                    print(f"✅ Uploaded to Blob: {file_url}")
            except ImportError as ie:
                 print("⚠️ vercel_blob not installed. Application Error.")
                 raise RuntimeError("Backend config error: vercel-blob invalid or missing. Check requirements.txt")
            except Exception as e:
                 print(f"⚠️ Blob Upload Failed: {e}")
                 # CRITICAL: If we are on Vercel, this is fatal regular persistence.
                 # Raise error so user knows upload failed.
                 raise RuntimeError(f"Vercel Blob Upload Failed: {str(e)}. Check BLOB_READ_WRITE_TOKEN.")

            # 4. Save to DB
            new_doc = Document(
                user_id=current_user.id, 
                filename=file.filename, 
                chunk_count=count,
                file_url=file_url # Save URL
            )
            db.session.add(new_doc)
            db.session.commit()
            
            log_activity(current_user.id, 'upload', file.filename)
            
            return jsonify({'message': 'File indexed successfully', 'chunks': count})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/create_session', methods=['POST'])
@login_required
def create_session():
    data = request.json
    title = data.get('title', 'New Research')
    doc_ids = data.get('doc_ids', [])
    
    # Create Session
    new_session = ChatSession(user_id=current_user.id, title=title)
    
    # Associate Documents
    if doc_ids:
        docs = Document.query.filter(Document.id.in_(doc_ids), Document.user_id == current_user.id).all()
        new_session.documents.extend(docs)
    
    db.session.add(new_session)
    db.session.commit()
    
    return jsonify({'session_id': new_session.id, 'message': 'Project created successfully'})

@app.route('/session/<int:session_id>', methods=['GET'])
@login_required
def get_session_details(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    
    return jsonify({
        'id': session.id,
        'title': session.title,
        'doc_count': len(session.documents),
        'doc_names': [d.filename for d in session.documents]
    })

@app.route('/session/<int:session_id>/documents', methods=['GET'])
@login_required
def get_session_documents(session_id):
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    docs = chat_session.documents
    
    return jsonify([{
        'id': d.id,
        'filename': d.filename,
        'uploaded_at': d.uploaded_at.isoformat() if d.uploaded_at else datetime.now().isoformat()
    } for d in docs])

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Setup Queue and Sentinel
    q = queue.Queue()
    job_done = object()

    # Callback Handler
    class QueueCallback(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs):
            q.put(token)

    try:
        data = request.json
        user_input = data.get('message')
        session_id = data.get('session_id')
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400

        # 1. Get Session
        if session_id:
            chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
        else:
            # Fallback or Create New (Legacy support)
            chat_session = ChatSession(user_id=current_user.id, title="General Chat")
            db.session.add(chat_session)
            db.session.commit()
        
        if not chat_session:
             return jsonify({'error': 'Session not found'}), 404

        # 2. Get History
        recent_msgs = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at.asc()).all()[-10:]
        history_str = ""
        for msg in recent_msgs:
            role_marker = "User" if msg.role == 'user' else "AI"
            history_str += f"{role_marker}: {msg.content}\n"
                
        # 3. Save User Message to DB immediately
        user_msg_db = ChatMessage(session_id=chat_session.id, role='user', content=user_input)
        db.session.add(user_msg_db)
        db.session.commit()

        # 4. Prepare Agent
        session_docs = chat_session.documents
        if session_docs:
            doc_names = [d.filename for d in session_docs]
            print(f"🔒 Context Isolated to Session Docs: {doc_names}")
        else:
            user_docs = Document.query.filter_by(user_id=current_user.id).all()
            doc_names = [d.filename for d in user_docs]
            print(f"🔓 No session docs linked. Fallback to all {len(doc_names)} user docs.")
        
        executor = get_agent_executor(current_user.id, doc_names=doc_names)
        
        # 5. Run Agent in Thread (Using RAG Graph)
        from rag_graph import rag_graph
        
        # Define retrieval function dynamically for the user
        def retrieval_fn(query):
            # Re-use the existing AstraDB setup logic or similar
            # Since we can't easily pass the entire retriever object to a compiled graph if it wasn't built in,
            # we pass the function via config.
            retriever = get_astradb_retriever(current_user.id)
            # Invoke retriever
            docs = retriever.invoke(query)
            return docs

        def task():
            try:
                # Prepare Config with user-specific retrieval
                config = {"configurable": {"user_id": str(current_user.id), "retrieval_fn": retrieval_fn}, "callbacks": [QueueCallback()]}
                
                # Invoke Graph
                # The graph state expects 'messages' or 'active_query'
                # We start with the user's input as the active query or message
                initial_state = {
                    "messages": [HumanMessage(content=user_input)], 
                    "active_query": user_input,
                    # Pass history if needed, but graph contextualizes it. 
                    # Actually, our graph expects 'messages' to be the history + new input.
                }
                
                # Append history to messages
                # We need to convert history_str or just rely on the new conversational flow?
                # The graph's 'contextualize' node takes all messages. 
                # Let's populate 'messages' with recent history + new input.
                
                # Fetch recent messages again as objects
                recent_msgs_objs = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at.asc()).all()[-5:]
                history_messages = []
                for m in recent_msgs_objs:
                    if m.role == 'user':
                        history_messages.append(HumanMessage(content=m.content))
                    else:
                        history_messages.append(AIMessage(content=m.content))
                
                # Add current input
                history_messages.append(HumanMessage(content=user_input))
                
                response = rag_graph.invoke(
                    {"messages": history_messages, "active_query": user_input},
                    config=config
                )
                
                # Extract Final Answer
                # The last message should be the AI response
                final_output = response["messages"][-1].content
                
                # Put the full result context in queue to be saved later
                q.put({'type': 'result', 'data': {'output': final_output}})

                # Put the full result context in queue to be saved later
                # q.put({'type': 'result', 'data': response}) 

            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put({'type': 'error', 'data': str(e)})
            finally:
                q.put(job_done)

        t = threading.Thread(target=task)
        t.start()
        
        # 6. Stream Generator
        def generate():
            full_streamed_content = "" # For tracking what was sent (optional)
            final_answer = "" # To be saved in DB
            
            while True:
                item = q.get()
                if item is job_done:
                    break
                
                if isinstance(item, dict):
                    if item.get('type') == 'result':
                        # Valid Result -> Extract output for DB
                        final_answer = item['data'].get('output', '')
                    elif item.get('type') == 'error':
                        # Send error to frontend
                        yield f"data: {json.dumps({'error': item['data']})}\n\n"
                        final_answer = f"Error: {item['data']}"
                else:
                    # Token
                    token = item
                    full_streamed_content += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Post-Streaming: Save AI Message to DB
            # Use final_answer from tool if available, else streamed content
            content_to_save = final_answer if final_answer else full_streamed_content

            # Fallback: If nothing was streamed but we have a final answer, send it now.
            if not full_streamed_content and content_to_save:
                 yield f"data: {json.dumps({'token': content_to_save})}\n\n"
            
            if content_to_save:
                # We need application context if not present, but stream_with_context handles request context? 
                # Actually, stream_with_context keeps the request context active.
                try:
                    ai_msg_db = ChatMessage(session_id=chat_session.id, role='ai', content=content_to_save, citations=json.dumps([]))
                    db.session.add(ai_msg_db)
                    db.session.commit()
                    log_activity(current_user.id, 'chat')
                except Exception as db_e:
                    print(f"DB Save Error: {db_e}")
            
            yield f"data: {json.dumps({'done': True})}\n\n"


        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as ie:
        import traceback
        trace = traceback.format_exc()
        print(f"CRITICAL CHAT ERROR: {trace}")
        return jsonify({
            'answer': f"System Error: {str(ie)}",
            'context': []
        })

@app.route('/pdf/<path:filename>')
@login_required
def serve_pdf(filename):
    """
    Securely serves the PDF file.
    1. Checks DB for Vercel Blob URL (Redirect).
    2. Fallback to local storage (for localhost or ephemeral).
    """
    safe_filename = os.path.basename(filename)
    
    # 1. Check DB for Blob URL
    # Exact Match
    doc = Document.query.filter_by(user_id=current_user.id, filename=safe_filename).order_by(Document.uploaded_at.desc()).first()
    
    # Fuzzy Match (Fallback for LLM typos like missing 'v7')
    if not doc:
        # Find any doc that contains the requested filename OR requested filename contains doc name
        all_docs = Document.query.filter_by(user_id=current_user.id).all()
        for d in all_docs:
            if safe_filename in d.filename or d.filename in safe_filename:
                doc = d
                safe_filename = d.filename # Update to real filename for local check
                print(f"📂 Fuzzy Match Found: Requested '{filename}' -> Found '{d.filename}'")
                break
    
    if doc and doc.file_url:
        print(f"📂 Redirecting to Blob: {doc.file_url}")
        return redirect(doc.file_url)

    # If we are seemingly on Vercel (doc exists but no URL and not found locally), guide the user
    # Note: We can't strictly detect Vercel here easily without env var, but the failure mode implies it.
    if doc and not doc.file_url:
        # Check if local file exists (for localhost support)
        user_prefixed_name = f"{current_user.id}_{doc.filename}"
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], user_prefixed_name)):
             return "⚠️ Storage Error: File URL missing. Please DELETE and RE-UPLOAD this document to save it to Vercel Blob.", 404

    # 2. Fallback: Local /tmp storage
    import glob
    
    # Check for specific user-prefixed file (using the Potentially Corrected safe_filename)
    user_prefixed_name = f"{current_user.id}_{safe_filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_prefixed_name)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    
    # Fallback: Try exact name
    file_path_exact = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    if os.path.exists(file_path_exact):
         return send_file(file_path_exact)
         
    return f"File not found. Searched for {user_prefixed_name} (Fuzzy corrected: {doc.filename if doc else 'None'})", 404

@app.route('/download_chat/<int:session_id>')
@login_required
def download_chat_export(session_id):
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    msgs = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
    
    # Generate Markdown/Text
    content = f"# Transkrip Bimbingan: {chat_session.title}\n"
    content += f"Tanggal: {datetime.now().strftime('%d %B %Y')}\n\n"
    
    for m in msgs:
        role = "Mahasiswa" if m.role == 'user' else "Dr. Sync (AI)"
        content += f"**{role}**:\n{m.content}\n\n---\n\n"
        
    from flask import Response
    return Response(
        content,
        mimetype="text/markdown",
        headers={"Content-disposition": f"attachment; filename=Draft_Skripsi_{session_id}.md"}
    )

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    # Helper to get specific session history
    session_id = request.args.get('session_id')
    if session_id:
        chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    else:
        chat_session = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).first()
        
    if not chat_session:
        return jsonify([])
        
    msgs = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at.asc()).all()
    
    output = []
    for m in msgs:
        citations = json.loads(m.citations) if m.citations else []
        output.append({
            'role': m.role,
            'content': m.content,
            'citations': citations
        })
    return jsonify(output)



@app.route('/document/<int:doc_id>/rename', methods=['PUT'])
@login_required
def rename_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    data = request.json
    new_name = data.get('filename')
    if new_name:
        # Note: This only changes the display name in the DB. 
        # Vector store metadata won't update, but user will see the new name in UI.
        doc.filename = new_name
        db.session.commit()
        return jsonify({'message': 'Document renamed'})
    return jsonify({'error': 'No filename provided'}), 400

@app.route('/document/<int:doc_id>/delete', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    
    try:
        # Optional: Delete from Vector Store here if possible
        # For now, we just delete the record so it can't be used for new chats.
        
        # Remove from database
        db.session.delete(doc)
        db.session.commit()
        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500



@app.route('/update_progress', methods=['POST'])
@login_required
def update_progress():
    data = request.json
    try:
        current_user.thesis_stage = int(data.get('stage', 0))
        db.session.commit()
        return jsonify({'message': 'Progress updated'})
    except:
        return jsonify({'error': 'Failed to update'}), 400

@app.route('/admin')
@app.route('/dashboard/admin') # Alias for user convenience
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Akses ditolak. Halaman ini hanya untuk Administrator.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Fetch Data
    users_count = User.query.count()
    docs_count = Document.query.count()
    chats_count = ChatSession.query.count()
    
    # Logs
    recent_logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(20).all()
    
    # Stats for Charts
    # import datetime # Removed to avoid shadowing
    today = datetime.now().date()
    actions_today = ActivityLog.query.filter(ActivityLog.timestamp >= today).count()
    
    login_count = ActivityLog.query.filter_by(action='login').count()
    upload_count = ActivityLog.query.filter_by(action='upload').count()
    chat_count = ActivityLog.query.filter_by(action='chat').count()

    stats = {
        'total_users': users_count,
        'total_docs': docs_count,
        'total_chats': chats_count,
        'actions_today': actions_today,
        'login_count': login_count,
        'upload_count': upload_count,
        'chat_count': chat_count
    }
    
    # Monthly Aggregation (Python-side for DB compatibility)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = {m: 0 for m in months}
    current_year = datetime.now().year
    
    # Optimization: If many logs, use SQL group_by. For now, simple iteration.
    all_logs = ActivityLog.query.all()
    for log in all_logs:
        if log.timestamp.year == current_year:
            m_name = log.timestamp.strftime('%b')
            if m_name in monthly_data:
                monthly_data[m_name] += 1

    return render_template('admin.html', stats=stats, recent_logs=recent_logs, monthly_data=monthly_data)

@app.route('/admin/reset-system', methods=['POST'])
@login_required
def reset_system():
    if not current_user.is_admin:
        flash('Unauthorized.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # 1. Delete Dependencies first (Logs, Messages)
        ActivityLog.query.delete()
        ChatMessage.query.delete()
        
        # 2. Delete Main Content (Sessions, Docs)
        # Note: session_documents association table rows are usually deleted via cascade if configured,
        # otherwise we might fail. Let's try deleting sessions first.
        ChatSession.query.delete()
        Document.query.delete()
        
        # 3. Delete Users (except current admin)
        User.query.filter(User.id != current_user.id).delete()
        
        # 4. Reset Current Admin Stats
        current_user.thesis_stage = 0
        
        db.session.commit()
        flash('SYSTEM RESET SUCCESSFUL: All users, documents, and logs have been deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"Reset Error: {e}")
        flash(f'Reset Failed: {e}', 'danger')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/export_logs')
@login_required
def export_logs():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
        
    logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).all()
    
    # Create CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'User ID', 'Action', 'Details', 'Timestamp'])
    
    for log in logs:
        cw.writerow([log.id, log.user_id, log.action, log.details, log.timestamp])
        
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    
    filename = f"activity_logs_{datetime.now().strftime('%Y%m%d')}.csv"
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/activity-stats')
@login_required
def get_activity_stats():
    # Last 7 days stats
    from datetime import timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    
    # Init dictionary for last 7 days
    dates = []
    counts = []
    
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        # Count logs for this date (User specific)
        # SQLAlchemy filter for date part only
        count = ActivityLog.query.filter(
            ActivityLog.user_id == current_user.id,
            db.func.date(ActivityLog.timestamp) == current.date()
        ).count()
        
        dates.append(current.strftime('%a')) # Mon, Tue
        counts.append(count)
        current += timedelta(days=1)
        
    return jsonify({
        'labels': dates,
        'data': counts
    })

@app.route('/generate_flashcards', methods=['POST'])
@login_required
def generate_flashcards():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
        
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return jsonify({'error': 'Session not found'}), 404
        
    # Get context from documents
    docs = session.documents
    if not docs:
         return jsonify({'error': 'No documents in this session to study.'}), 400
         
    doc_context = ", ".join([d.filename for d in docs])
    
    # Minimal RAG for Flashcards (using same vector store or just filenames? 
    # Ideally should retrieve actual content, but for speed/token limits without complex RAG in this specific fn, 
    # we might just ask for general concepts if content not retrieved. 
    # BETTER: Retrieve a summary or random chunks. 
    # SIMPLIFIED APPROACH for SPEED: Use the 'get_agent_executor' but with a specific prompt
    # OR: Just separate LLM call. Let's use the simplest valid approach:
    # We will trigger the existing agent with a specific instruction.)
    
    # Re-using the Agent Executor is robust because it has the tools.
    executor = get_agent_executor(current_user.id, [d.filename for d in docs])
    
    prompt = f"""
    Generasikan 5 Flashcard belajar (Kartu Kilat) berdasarkan dokumen-dokumen yang ada di sesi ini.
    Fokus pada konsep kunci, definisi, atau argumen utama dalam skripsi/dokumen tersebut.
    
    Format Output WAJIB JSON murni seperti ini (tanpa markdown blok):
    {{
        "flashcards": [
            {{ "question": "Apa itu X?", "answer": "X adalah..." }},
            {{ "question": "Mengapa Y penting?", "answer": "Karena..." }}
        ]
    }}
    
    JANGAN memberikan teks lain selain JSON tersebut.
    """
    
    try:
        response = executor.invoke({"input": prompt, "chat_history": ""})
        output_text = response['output']
        
        # Cleanup potential markdown around JSON
        output_text = output_text.strip()
        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").replace("```", "")
        elif output_text.startswith("```"):
             output_text = output_text.replace("```", "")
             
        return jsonify(json.loads(output_text))
    except Exception as e:
        print(f"Flashcard Gen Error: {e}")
        # Fallback dummy if AI fails
        return jsonify({
            "flashcards": [
                { "question": "Error generating", "answer": str(e) }
            ]
        })

@app.route('/api/audio_summary/<int:session_id>')
@login_required
def get_audio_summary(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return "Session not found", 404
        
    # Logic: Get the last AI message
    last_msg = ChatMessage.query.filter_by(session_id=session.id, role='ai').order_by(ChatMessage.created_at.desc()).first()
    
    text_to_speak = "Belum ada percakapan untuk diringkas."
    
    if last_msg:
        # Clean markdown
        import re
        clean_text = re.sub(r'[*_#`\[\]]', '', last_msg.content)
        # Limit length for speed
        text_to_speak = clean_text[:500] + ("..." if len(clean_text) > 500 else "")
        text_to_speak = "Berikut ringkasan audio dari jawaban terakhir: " + text_to_speak
    else:
        text_to_speak = "Halo! Ini adalah ScholarSync. Mulailah percakapan untuk mendengarkan ringkasan."

    try:
        # Generate Audio
        tts = gTTS(text_to_speak, lang='id')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return send_file(
            fp,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='summary.mp3'
        )
    except Exception as e:
        print(f"TTS Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    with app.app_context():
        # Ensure migration for new table
        db.create_all()
        
        # Create Default Admin if not exists
        try:
            if not User.query.filter_by(username='admin').first():
                print("Auto-creating default admin...")
                hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
                admin = User(username='admin', email='admin@scholarsync.com', password=hashed_password, is_admin=True)
                db.session.add(admin)
                db.session.commit()
                print("Default admin created.")
        except Exception as e:
            print(f"Admin creation failed (might be connection issue): {e}")
            
    app.run(debug=True, port=8501)
