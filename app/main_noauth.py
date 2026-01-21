from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers.portfolios_noauth import router as portfolios_router
from app.routers.assets_noauth import router as assets_router
from app.routers.recommendations_noauth import router as recommendations_router
from app.routers.preferences_noauth import router as preferences_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Investment Portfolio Tracker API",
    description="Real-time portfolio tracking with ML-powered recommendations (No Auth)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolios_router)
app.include_router(assets_router)
app.include_router(recommendations_router)
app.include_router(preferences_router)

@app.get("/")
def read_root():
    return {
        "message": "Investment Portfolio Tracker API",
        "version": "1.0.0",
        "docs": "/docs",
        "auth": "disabled"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "version": "1.0.0",
        "auth": "disabled"
    }
