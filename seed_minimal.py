
import os
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
# We import models inside to safely map them, or redefine them if imports fail due to env
# Better to redefine minimal models if just wiping/seeding User
# But we nneed to wipe ALL tables.
# Let's try importing from flask_app.models

from flask_app.models import db, User, ChatSession, ChatMessage, Document, ActivityLog

# Define the app
app = Flask(__name__)

# Use the Neon DB URL directly from the .env found
db_url = "postgresql://neondb_owner:npg_w9dnCiaKzZW1@ep-curly-bar-ada0vhuc-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Fix postgres protocol
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)

def seed_admin():
    with app.app_context():
        print("Connecting to Neon Database...")
        # Drop and Recreate All Tables (Reset to 0)
        db.drop_all()
        print("Tables dropped.")
        db.create_all()
        print("Tables created.")
        
        # Create Admin
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(
            username='admin',
            email='admin@scholarsync.com',
            password=hashed_password,
            is_admin=True,
            thesis_stage=0
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ SUCCESS: Admin user created.")
        print("Email: admin@scholarsync.com")
        print("Password: admin123")

if __name__ == "__main__":
    seed_admin()
