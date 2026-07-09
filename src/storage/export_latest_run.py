import sqlite3

conn = sqlite3.connect("data/race_results.db")
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