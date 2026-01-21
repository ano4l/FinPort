from sqlalchemy import create_engine
from app.database import Base
from app.models import User, Portfolio, Asset, Holding, Transaction, PriceHistory, Recommendation, UserPreferences
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/portfolio_tracker")

def init_database():
    print("Creating database tables...")
    print(f"Connecting to: {DATABASE_URL[:50]}...")
    
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    
    print("✅ Database tables created successfully!")
    print("\nTables created:")
    print("- users")
    print("- user_preferences")
    print("- portfolios")
    print("- assets")
    print("- holdings")
    print("- transactions")
    print("- price_history")
    print("- recommendations")

if __name__ == "__main__":
    init_database()
