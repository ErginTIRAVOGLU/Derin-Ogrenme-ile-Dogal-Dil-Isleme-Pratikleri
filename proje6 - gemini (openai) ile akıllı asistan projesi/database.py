import sqlite3
import os

# veritabanı yolu
DB_PATH = os.path.join("data","assistant.db")

# veritabanı bağlantısı aç
def initialize_db():
    os.makedirs("data", exist_ok=True) # veritabanı klasörünü oluştur
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # tablo oluştur
    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    ''')
    
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS calendar (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       event TEXT NOT NULL,
                       event_date TEXT NOT NULL
                   )
                   ''')
    
    conn.commit()
    conn.close()

# not ekle
def add_note(content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))
    
    conn.commit()
    conn.close()

# etkinlik ekle
def add_event(event, event_date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO calendar (event, event_date) VALUES (?, ?)", (event, event_date))
    
    conn.commit()
    conn.close()

# tüm notları getir
def get_notes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT content, created_at FROM notes ORDER BY created_at DESC")
    notes = cursor.fetchall()
    
    conn.close()
    return notes

# tüm etkinlikleri getir
def get_events():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT event, event_date FROM calendar ORDER BY event_date")
    events = cursor.fetchall()
    
    conn.close()
    return events
            

    
if __name__ == "__main__":
    initialize_db()
    # add_note("eve dönerken su almayı unutma")
    # add_event("toplantı var", "15.12.2027")
    
    # print(f"Notlar: {get_notes()}")
    # print(f"Etkinlikler: {get_events()}")