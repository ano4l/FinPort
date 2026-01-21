from app.database import Base, engine
from app.models import User, Portfolio, Asset, Holding, Transaction, PriceHistory, Recommendation, UserPreferences

def init_database():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
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
