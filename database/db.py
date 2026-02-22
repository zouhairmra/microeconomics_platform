import sqlite3
import os

DB_PATH = "database/database.db"

def connect():
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_tables():

    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS scores(
        username TEXT,
        game TEXT,
        score REAL
    )
    """)

    conn.commit()
    conn.close()

def add_user(username):

    conn = connect()
    c = conn.cursor()

    c.execute("INSERT OR IGNORE INTO users VALUES(?)",(username,))

    conn.commit()
    conn.close()

def save_score(username,game,score):

    conn = connect()
    c = conn.cursor()

    c.execute("INSERT INTO scores VALUES(?,?,?)",(username,game,score))

    conn.commit()
    conn.close()

def get_scores():

    conn = connect()
    c = conn.cursor()

    c.execute("SELECT * FROM scores")

    data = c.fetchall()

    conn.close()

    return data
