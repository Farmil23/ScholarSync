
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    thesis_stage = db.Column(db.Integer, default=0) # 0: Judul, 1: Bab 1, 2: Bab 2, 3: Bab 3, 4: Sidang

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Nullable for anonymous visits if needed
    action = db.Column(db.String(50), nullable=False) # e.g., 'login', 'upload', 'chat'
    details = db.Column(db.String(255), nullable=True) # e.g., filename, metadata
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Association Table for Many-to-Many (Session <-> Document)
session_documents = db.Table('session_documents',
    db.Column('session_id', db.Integer, db.ForeignKey('chat_session.id'), primary_key=True),
    db.Column('document_id', db.Integer, db.ForeignKey('document.id'), primary_key=True)
)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Conversation")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade="all, delete-orphan")
    documents = db.relationship('Document', secondary=session_documents, lazy='subquery',
        backref=db.backref('sessions', lazy=True))

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False) # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Optional: Store citations as JSON string if needed
    citations = db.Column(db.Text, nullable=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_url = db.Column(db.String(512), nullable=True) # Vercel Blob URL (or S3)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    # chunk_count could be useful
    chunk_count = db.Column(db.Integer, default=0) 
