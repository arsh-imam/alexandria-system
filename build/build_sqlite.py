"""One-time: convert passages.jsonl to SQLite for fast random access."""
import sqlite3, json, time, os

JSONL = "/media/pi/KINGSTON/local_ai/index/passages.jsonl"
DB    = "/media/pi/KINGSTON/local_ai/index/passages.db"

def main():
    if os.path.exists(DB):
        os.remove(DB)

    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("""CREATE TABLE passages (
        id INTEGER PRIMARY KEY, title TEXT, text TEXT, source TEXT)""")

    print("Converting passages to database...", flush=True)
    t = time.time()
    batch = []
    with open(JSONL, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            batch.append((i, obj["title"], obj["text"], obj["source"]))
            if len(batch) >= 10000:
                db.executemany("INSERT INTO passages VALUES (?,?,?,?)", batch)
                batch = []
                if i % 200000 < 10000:
                    print(f"  {i:,} rows...", flush=True)
    if batch:
        db.executemany("INSERT INTO passages VALUES (?,?,?,?)", batch)
    db.commit()

    print("Creating index...", flush=True)
    db.execute("CREATE INDEX idx_source ON passages(source)")
    db.commit()
    db.close()

    elapsed = time.time() - t
    size_mb = os.path.getsize(DB) / 1024 / 1024
    print(f"Done in {elapsed:.0f}s. Database: {size_mb:.0f} MB", flush=True)

if __name__ == '__main__':
    main()
