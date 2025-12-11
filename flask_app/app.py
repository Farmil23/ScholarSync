
import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime

# LangChain Imports
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings # Removed for Vercel

# Local Imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_llm
from ingest_handler import ingest_file
from flask_app.models import db, User, ChatSession, ChatMessage, Document, ActivityLog

app = Flask(__name__, instance_path='/tmp', instance_relative_config=True)
# Config
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Database Config for Vercel (Neon/Postgres)
# Database Config for Vercel (Neon/Postgres)
database_url = os.getenv("DATABASE_URL")
if not database_url:
    # CRITICAL: In serverless (Vercel), we CANNOT use SQLite/Memory for persistence.
    # We must enforce Neon/Postgres.
    print("❌ ERROR: DATABASE_URL is missing in environment variables.")
    raise RuntimeError("DATABASE_URL is missing. Please add it in Vercel Settings.")

if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp/uploads' # Use /tmp which is writable on Vercel

# Init Extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Setup Uploads
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
    except Exception as e:
        print(f"⚠️ Database connection failed during startup: {e}")
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

def get_agent_executor(user_id):
    llm = get_llm()
    
    # 1. Tools
    retriever = get_astradb_retriever(user_id)
    retriever_tool = create_retriever_tool(
        retriever,
        "search_uploaded_documents",
        "Searches and returns answers from the documents uploaded by the user. Use this FIRST if the user asks about their files."
    )
    
    search_tool = DuckDuckGoSearchRun(
        name="web_search", 
        description="Search the web for general knowledge, current events, or info not in the documents."
    )
    
    tools = [retriever_tool, search_tool]
    
    # 2. React Prompt
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    
    # 3. Create Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 4. Create Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

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

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Access Denied", "danger")
        return redirect(url_for('index'))
    
    # Calculate Stats
    total_users = User.query.count()
    total_docs = Document.query.count()
    total_chats = ChatMessage.query.filter_by(role='user').count()
    
    today = datetime.utcnow().date()
    actions_today = ActivityLog.query.filter(ActivityLog.timestamp >= today).count()
    
    login_count = ActivityLog.query.filter_by(action='login').count()
    upload_count = ActivityLog.query.filter_by(action='upload').count()
    chat_count = ActivityLog.query.filter_by(action='chat').count()
    
    recent_logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(10).all()
    
    stats = {
        'total_users': total_users,
        'total_docs': total_docs,
        'total_chats': total_chats,
        'actions_today': actions_today,
        'login_count': login_count,
        'upload_count': upload_count,
        'chat_count': chat_count
    }
    
    return render_template('admin.html', stats=stats, recent_logs=recent_logs)

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
                
            return redirect(url_for('chat_page'))
        else:
            flash('Login Failed. Check email and password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'warning')
            return redirect(url_for('register'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Auto-Admin for specific email
        is_admin = (email == 'admin@scholarsync.com')
        
        new_user = User(username=username, email=email, password=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        log_activity(new_user.id, 'register')
        return redirect(url_for('chat_page'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html', user=current_user)

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_{file.filename}")
        file.save(filepath)
        
        try:
            count = ingest_file(filepath, user_id, file.filename)
            os.remove(filepath)
            if count == 0:
                 return jsonify({'error': 'Could not extract text from PDF'}), 400
            
            new_doc = Document(user_id=current_user.id, filename=file.filename, chunk_count=count)
            db.session.add(new_doc)
            db.session.commit()
            
            log_activity(current_user.id, 'upload', file.filename)
            
            return jsonify({'message': 'File indexed successfully', 'chunks': count})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({'error': 'Empty message'}), 400

    # 1. Get or Create Session
    session_title = "General Chat"
    chat_session = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).first()
    if not chat_session:
        chat_session = ChatSession(user_id=current_user.id, title=session_title)
        db.session.add(chat_session)
        db.session.commit()
    
    # 2. Get History (Last 5 turns)
    recent_msgs = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at.asc()).all()[-10:]
    
    history_str = ""
    for msg in recent_msgs:
        role_marker = "User" if msg.role == 'user' else "AI"
        history_str += f"{role_marker}: {msg.content}\n"
            
    # 3. Invoke Agent
    executor = get_agent_executor(current_user.id)
    
    try:
        response = executor.invoke({
            "input": user_input,
            "chat_history": history_str
        })
        answer = response["output"]
    except Exception as e:
        print(f"Agent Execution Error: {e}")
        # Build fallback if agent fails (e.g. parsing error loop)
        answer = "I apologize, I encountered an issue while searching. Please try again."

    # 4. Save to DB
    user_msg_db = ChatMessage(session_id=chat_session.id, role='user', content=user_input)
    ai_msg_db = ChatMessage(session_id=chat_session.id, role='ai', content=answer, citations=json.dumps([]))
    
    db.session.add(user_msg_db)
    db.session.add(ai_msg_db)
    db.session.commit()
    
    log_activity(current_user.id, 'chat')
    
    return jsonify({
        'answer': answer,
        'context': []
    })

@app.route('/history', methods=['GET'])
@login_required
def get_history():
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

if __name__ == '__main__':
    app.run(debug=True, port=8501)
