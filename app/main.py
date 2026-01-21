from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import portfolios, assets, recommendations, preferences, advanced_analytics

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Investment Portfolio Tracker API",
    description="Real-time portfolio tracking with ML-powered recommendations and advanced quantitative analytics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolios.router)
app.include_router(assets.router)
app.include_router(recommendations.router)
app.include_router(preferences.router)
app.include_router(advanced_analytics.router)

@app.get("/")
def read_root():
    return {
        "message": "Investment Portfolio Tracker API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
