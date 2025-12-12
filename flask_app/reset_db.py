
from app import app, db
from models import User, ChatSession, ChatMessage, Document, ActivityLog

def reset_database():
    with app.app_context():
        # Option 1: Drop All and Recreate (Cleanest)
        print("Dropping all tables...")
        db.drop_all()
        print("Creating all tables...")
        db.create_all()

        # Create Default Admin
        from flask_bcrypt import Bcrypt
        bcrypt = Bcrypt(app)
        # Check if hash needed
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
        
        print("Database reset complete. Default admin created (email: admin@scholarsync.com, pass: admin123).")

if __name__ == "__main__":
    reset_database()
