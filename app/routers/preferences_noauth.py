from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter(prefix="/api/preferences", tags=["preferences"])

class UserPreferences(BaseModel):
    risk_tolerance: str = "medium"
    preferred_assets: list = []
    investment_horizon_years: int = 5
    auto_rebalance: bool = False
    notification_enabled: bool = True
    theme: str = "dark"

# Mock preferences (since no auth)
MOCK_PREFERENCES = UserPreferences()

@router.get("/")
def get_user_preferences():
    """Get mock user preferences"""
    return MOCK_PREFERENCES

@router.put("/")
def update_user_preferences(preferences: UserPreferences):
    """Update mock user preferences"""
    global MOCK_PREFERENCES
    MOCK_PREFERENCES = preferences
    return {"message": "Preferences updated successfully", "preferences": MOCK_PREFERENCES}
