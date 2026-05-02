import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "users.db"

def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def get_or_create_user(phone_number: str) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Try to find the user
    cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
    user = cursor.fetchone()
    
    now = datetime.now(timezone.utc).isoformat()

    if user:
        # Update last login
        cursor.execute("UPDATE users SET last_login = ? WHERE phone_number = ?", (now, phone_number))
        conn.commit()
        user_dict = dict(user)
        user_dict['last_login'] = now
    else:
        # Create new user
        cursor.execute(
            "INSERT INTO users (phone_number, created_at, last_login) VALUES (?, ?, ?)",
            (phone_number, now, now)
        )
        conn.commit()
        
        cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
        user_dict = dict(cursor.fetchone())
        
    conn.close()
    return user_dict
