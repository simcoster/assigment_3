"""Persistent LangGraph checkpoint storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_checkpointer(db_path: Path) -> SqliteSaver:
    """Create a SQLite-backed checkpointer for conversation session persistence."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
