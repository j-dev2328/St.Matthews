import sqlite3

db_path = 'anpr.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# plates table — now includes name and vehicle_type
cursor.execute("""
CREATE TABLE IF NOT EXISTS plates (
    plate TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    vehicle_type TEXT DEFAULT ''
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT,
    timestamp TEXT,
    status TEXT,
    direction TEXT DEFAULT ''
)
""")

conn.commit()

# Add new columns if upgrading from old database (safe to run even if columns exist)
for col in ["name TEXT DEFAULT ''", "vehicle_type TEXT DEFAULT ''"]:
    try:
        cursor.execute(f"ALTER TABLE plates ADD COLUMN {col}")
        conn.commit()
    except:
        pass

# Add direction column to logs if upgrading from old database
try:
    cursor.execute("ALTER TABLE logs ADD COLUMN direction TEXT DEFAULT ''")
    conn.commit()
except:
    pass


def load_plates():
    cursor.execute("SELECT plate FROM plates")
    return [row[0] for row in cursor.fetchall()]
