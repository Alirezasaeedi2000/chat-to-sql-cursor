"""
Database connection and utility functions.
Extracted from query_processor.py for better modularity.
"""

import os
import logging
from typing import Any, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create necessary output directories."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/exports", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/cache", exist_ok=True)


def create_engine_from_url(db_url: str) -> Engine:
    """Create SQLAlchemy engine from database URL with optimized settings."""
    connect_args: Dict[str, Any] = {}
    
    if db_url.startswith("mysql+"):
        connect_args["connect_timeout"] = 10
        connect_args["charset"] = "utf8mb4"
    elif db_url.startswith("postgresql+"):
        connect_args["connect_timeout"] = 10
    elif db_url.startswith("sqlite+"):
        # SQLite specific optimizations
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
        echo=False,  # Set to True for SQL debugging
    )
    return engine


def create_engine_from_env() -> Engine:
    """Create engine from DATABASE_URL environment variable with error handling."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    # Test the connection with better error messaging
    try:
        engine = create_engine_from_url(db_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        if "Access denied" in str(e):
            raise RuntimeError(
                f"Database authentication failed. Please check your DATABASE_URL credentials: {e}"
            )
        elif "Can't connect" in str(e) or "Connection refused" in str(e):
            raise RuntimeError(
                f"Cannot connect to database server. Please verify the host and port in DATABASE_URL: {e}"
            )
        elif "Unknown database" in str(e):
            raise RuntimeError(
                f"Database does not exist. Please check the database name in DATABASE_URL: {e}"
            )
        else:
            raise RuntimeError(f"Database connection failed: {e}")
