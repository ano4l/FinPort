"""Vercel serverless function entry point"""
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and initialize the FastAPI app
from app.main import app
from app.database import Base, engine

# Initialize database tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Database initialization warning: {e}")

# Export the app for Vercel



