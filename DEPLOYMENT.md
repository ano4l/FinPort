# Portfolio Tracker - Deployment Guide

## Overview
This is a FastAPI-based investment portfolio tracker with a Next.js frontend. Authentication has been completely removed for simplified deployment.

## Local Development

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python test_server.py
```
Backend runs on: `http://localhost:8000`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on: `http://localhost:3002`

## Vercel Deployment

### Backend Deployment
The backend is configured for Vercel deployment with:
- `vercel.json` - Vercel configuration
- `api/index.py` - Entry point for Vercel
- Minimal `requirements.txt` with only essential dependencies

### Environment Variables Required
- `DATABASE_URL` - SQLite or PostgreSQL connection string
- `REDIS_URL` - Redis connection (optional)

### Deploy Commands
```bash
# From backend directory
vercel deploy
```

### Frontend Deployment
The frontend can be deployed to Vercel directly:
```bash
# From frontend directory
vercel deploy
```

## Database
- Local: SQLite (`portfolio_tracker.db`)
- Production: Configure `DATABASE_URL` environment variable

## API Documentation
Once running, visit: `http://localhost:8000/docs`

## Features
- Real-time portfolio tracking
- Multiple asset classes (stocks, crypto, ETFs, etc.)
- No authentication required
- Sample data included
