"""
Builds an FTS5 (BM25) full-text index over all 2,028,337 passages,
replacing the unindexed LIKE '%x%' title scans (4-17s each, junk-prone).
External-content table: stores only the inverted index (~2-3GB),
text stays in the existing passages table. Reversible:
  DROP TABLE passages_fts;  undoes everything.
"""
import os, re, sys, time, sqlite3

DB = "/media/pi/KINGSTON/local_ai/index/passages.db"
BATCH = 25000

# FTS5 available?
try:
    sqlite3.connect(":memory:").execute(
        "CREATE VIRTUAL TABLE t USING fts5(x)")
except Exception as e:
    print(f"FTS5 not available in this sqlite3 build: {e}"); sys.exit(1)

st = os.statvfs(os.path.dirname(DB))
free_gb = st.f_bavail * st.f_frsize / 1e9
print(f"Free space on SSD: {free_gb:.1f} GB (need ~4 GB)")
if free_gb < 4:
    print("Not enough free space — stopping."); sys.exit(1)

db = sqlite3.connect(DB)
db.execute("PRAGMA synchronous=OFF")
db.execute("PRAGMA cache_size=-200000")
total = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
print(f"Passages: {total:,}")

print("Dropping old FTS table if present...")
db.execute("DROP TABLE IF EXISTS passages_fts")
db.commit()

print("Creating FTS5 table (porter stemming, external content)...")
db.execute("""
    CREATE VIRTUAL TABLE passages_fts USING fts5(
        title, text,
        content='passages', content_rowid='id',
        tokenize='porter unicode61'
    )""")
db.commit()

print(f"Indexing {total:,} passages in batches of {BATCH:,}...\n")
start = time.time()
done = 0
while done < total:
    rows = db.execute(
        "SELECT id, title, text FROM passages "
        "WHERE id >= ? AND id < ?", (done, done + BATCH)).fetchall()
    db.executemany(
        "INSERT INTO passages_fts(rowid, title, text) VALUES (?,?,?)",
        rows)
    db.commit()
    done += BATCH
    d = min(done, total)
    el = time.time() - start
    rate = d / el if el else 0
    eta = (total - d) / rate / 60 if rate else 0
    print(f"  [{d/total*100:5.1f}%] {d:,}/{total:,} | "
          f"{rate:.0f}/s | ETA {eta:.0f} min", flush=True)

print("\nOptimizing FTS index (can take a few minutes)...")
db.execute("INSERT INTO passages_fts(passages_fts) VALUES('optimize')")
db.commit()
db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
size_gb = os.path.getsize(DB) / 1e9
print(f"Done in {(time.time()-start)/60:.1f} min. DB now {size_gb:.1f} GB.")

# ── Sanity: the queries the old title search failed on ──────────────
STOP = {'the','a','an','is','are','was','were','what','who','why','how',
        'does','do','did','of','in','on','to','and','or','it','between',
        'mean','means'}

def fts_query(q):
    terms = [t for t in re.findall(r"[a-zA-Z0-9]+", q.lower())
             if t not in STOP and len(t) >= 2]
    return " OR ".join(f'"{t}"' for t in terms)

print("\n" + "=" * 60)
print("SANITY — BM25 top-5 (title weighted 10x), with timing")
print("=" * 60)
for q in ["what causes the seasons",
          "what is the difference between a virus and bacteria",
          "what does ephemeral mean",
          "what is the basic structure doctrine",
          "why is the sky blue",
          "who was Andrei Tarkovsky"]:
    m = fts_query(q)
    t0 = time.time()
    rows = db.execute(
        "SELECT rowid, bm25(passages_fts, 10.0, 1.0) AS r "
        "FROM passages_fts WHERE passages_fts MATCH ? "
        "ORDER BY r LIMIT 5", (m,)).fetchall()
    dt = (time.time() - t0) * 1000
    print(f"\nQ: {q}   ({dt:.0f} ms)")
    for rid, r in rows:
        row = db.execute(
            "SELECT title, source FROM passages WHERE id=?",
            (rid,)).fetchone()
        print(f"    [{r:8.2f}] {row[1]:<10} {row[0][:48]!r}")
print("\nFTS5 build complete.")
