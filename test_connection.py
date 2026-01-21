"""Test Supabase connection"""
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print(f"Testing connection to: {DATABASE_URL[:50]}...")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Successfully connected to Supabase!")
        print(f"✅ Database is responding")
except Exception as e:
    print(f"❌ Connection failed: {e}")
