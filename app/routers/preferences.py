from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import UserPreferences
from app.schemas import UserPreferences as UserPreferencesSchema, UserPreferencesCreate

router = APIRouter(prefix="/api/preferences", tags=["User Preferences"])

@router.get("/", response_model=UserPreferencesSchema)
def get_preferences(
    db: Session = Depends(get_db)
):
    preferences = db.query(UserPreferences).first()
    
    if not preferences:
        preferences = UserPreferences()
        db.add(preferences)
        db.commit()
        db.refresh(preferences)
    
    return preferences

@router.put("/", response_model=UserPreferencesSchema)
def update_preferences(
    preferences_update: UserPreferencesCreate,
    db: Session = Depends(get_db)
):
    preferences = db.query(UserPreferences).first()
    
    if not preferences:
        preferences = UserPreferences()
        db.add(preferences)
    
    update_data = preferences_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(preferences, field, value)
    
    db.commit()
    db.refresh(preferences)
    return preferences
