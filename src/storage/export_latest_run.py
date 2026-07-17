import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "generated" / "race_results.db"

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("\nTables:\n")

for table in tables:
    name = table[0]
    print("=" * 60)
    print(name)
    print("=" * 60)

    cursor.execute(f"PRAGMA table_info({name})")
    for column in cursor.fetchall():
        print(column)

    print()

conn.close()
