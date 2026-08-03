"""
db.py
=====
Lightweight SQLite persistence for Masar AI.

- users        : one row per account (hashed password, never plaintext)
- logs         : an append-only activity trail (logins, uploads, predictions...)

The database lives in masar_ai.db next to app.py. On a host with a
persistent filesystem (your own server, a VM, Docker with a volume) this
survives restarts. On an ephemeral host (some free hosting tiers wipe the
filesystem on every redeploy) it will reset — swap DB_PATH for a managed
database (Postgres, Supabase, etc.) if you need it to survive redeploys too.
"""

import os
import sqlite3
import hashlib
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masar_ai.db")

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "masar2026"  # seeded once on first run; change the account's
                               # password afterwards from a real admin panel
                               # rather than editing this constant long-term.


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    if conn.execute("SELECT 1 FROM users WHERE username=?", (ADMIN_USERNAME,)).fetchone() is None:
        _create_user(conn, ADMIN_USERNAME, ADMIN_PASSWORD)
    conn.close()


def _create_user(conn, username, password):
    salt = os.urandom(8).hex()
    conn.execute(
        "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?,?,?,?)",
        (username, _hash(password, salt), salt, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()


def user_exists(username: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return row is not None


def add_user(username: str, password: str):
    conn = get_conn()
    _create_user(conn, username, password)
    conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row is None:
        return False
    return _hash(password, row["salt"]) == row["password_hash"]


def touch_login(username: str):
    conn = get_conn()
    conn.execute("UPDATE users SET last_login=? WHERE username=?",
                 (datetime.datetime.utcnow().isoformat(), username))
    conn.commit()
    conn.close()


def log_event(username: str, action: str, details: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO logs (username, action, details, timestamp) VALUES (?,?,?,?)",
        (username, action, details, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_users():
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT username, created_at, last_login FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df


def get_all_logs(limit: int = 500):
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, timestamp, username, action, details FROM logs ORDER BY id DESC LIMIT ?",
        conn, params=(limit,))
    conn.close()
    return df
