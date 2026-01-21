import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import Base, engine

# Create tables on startup
Base.metadata.create_all(bind=engine)

# Export app for Vercel
__all__ = ['app']

