"""WSGI entry point for Vercel deployment"""
from app.main import app
from app.database import Base, engine

# Create tables on startup
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not create tables: {e}")

# Export for Vercel
application = app
