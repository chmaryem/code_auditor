"""
database/ — Persistent storage layer (PostgreSQL via SQLAlchemy 2.x).

Public surface:
    from database import engine, AsyncSession, get_db
    from database.models import User, Conversation, Message, ...
    from database.repositories import UserRepo, ConversationRepo, ...
"""
from database.connection import engine, AsyncSession, get_db

__all__ = ["engine", "AsyncSession", "get_db"]
