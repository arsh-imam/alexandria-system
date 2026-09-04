"""
Targeted extraction with pageview-ranked Wikipedia + capped/full small sources.
Resumable at source level AND within Wikipedia (every 100k entries).
Dual-writes to SSD + local disk for crash safety.
"""

import os
import re
import json
import time
import unicodedata

SSD            = r"G:\local_ai"
ZIM_DIR        = os.path.join(SSD, "wikipedia")
INDEX_DIR      = os.path.join(SSD, "index")
TITLES_FILE    = os.path.join(INDEX_DIR, "top_200k_titles.txt")
PASSAGES_FILE  = os.path.join(INDEX_DIR, "passages.jsonl")
PROGRESS_FILE  = os.path.join(INDEX_DIR, "extract_progress.json")

LOCAL_DIR      = os.path.join(os.path.expanduser("~"), "lcatrs_extract")
LOCAL_PASSAGES = os.path.join(LOCAL_DIR, "passages.jsonl")

# (filename, tag, use_title_filter, min_chars, max_passages, max_articles)
# max_articles=None means take all. Wiktionary capped to avoid explosion.
SOURCES = [
    ("wikinews_en.zim",   "wikinews",   False,  500, 2, None),
    ("wikiquote_en.zim",  "wikiquote",  False,  300, 2, None),
    ("wikivoyage_en.zim", "wikivoyage", False,  600, 5, None),
    ("wiktionary_en.zim", "wiktionary", False,  500, 1, 40000),
    ("ifixit_en.zim",     "ifixit",     False,  500, 5, None),
    ("wikibooks_en.zim",  "wikibooks",  False, 1000, 6, None),
    ("wikipedia_en.zim",  "wikipedia",  True,  2000, 6, None),
]

WORDS_PER_PASSAGE     = 160
OVERLAP_WORDS         = 40
WIKI_CHECKPOINT_EVERY = 100000


def normalize(title):
    t = unicodedata.normalize('NFC', title).lower().strip()
    t = re.sub(r'\s+', ' ', t).replace('_', ' ')
    return t


def load_title_set(filepath):
    titles = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                titles[normalize(parts[1])] = (parts[1], int(parts[0]))
    print(f"Loaded {len(titles):,} titles from title set.", flush=True)
    return titles


def clean_html(raw):
    raw = re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=re.S)
    raw = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.S)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'&[a-zA-Z#0-9]+;', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw)
    return raw.strip()


def chunk_overlapping(text, max_passages):
    words = text.split()
    chunks = []
    step = WORDS_PER_PASSAGE - OVERLAP_WORDS
    for i in range(0, len(words), step):
        piece = words[i : i + WORDS_PER_PASSAGE]
        if len(piece) > 40:
            chunks.append(' '.join(piece))
        if len(chunks) >= max_passages:
            break
    return chunks


def looks_like_junk(title):
    t = title.lower()
    return any(b in t for b in [
        'disambiguation', 'list of', 'index of',
        '(disambiguation)', 'template:', 'category:',
        'file:', 'portal:', 'wikipedia:', 'help:'
    ])


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed_sources": [],
        "total_passages": 0,
        "wikipedia_resume_from": 0,
        "wikipedia_passages": 0,
        "wikipedia_matched": 0,
    }


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def extract_wikipedia(title_set, out, progress):
    from libzim.reader import Archive

    path = os.path.join(ZIM_DIR, "wikipedia_en.zim")
    if not os.path.exists(path):
        print(f"  SKIP wikipedia — not found.", flush=True)
        return 0

    archive   = Archive(path)
    total     = archive.all_entry_count
    get_entry = getattr(archive, 'get_entry_by_id',
                getattr(archive, '_get_entry_by_id', None))

    resume_from = progress["wikipedia_resume_from"]
    matched     = progress["wikipedia_matched"]
    passages    = progress["wikipedia_passages"]

    print(f"\n=== WIKIPEDIA ===", flush=True)
    print(f"  Entries: {total:,}  |  resume from: {resume_from:,}", flush=True)
    print(f"  Already matched: {matched:,}  passages: {passages:,}", flush=True)

    start = time.time()
    last_checkpoint = resume_from
    probe_done = False

    for i in range(resume_from, total):
        try:
            entry = get_entry(i)
            if entry.is_redirect: continue
            title = str(entry.title).strip()
            if not title or looks_like_junk(title): continue
            seg = str(entry.path).split('/')[-1]
            if '.' in seg and not seg.endswith('.html'): continue

            norm = normalize(title)
            if norm not in title_set: continue

            raw  = bytes(entry.get_item().content).decode('utf-8', errors='ignore')
            text = clean_html(raw)
            if len(text) < 2000: continue

            _, views = title_set.get(norm, (title, 0))

            for ptext in chunk_overlapping(text, 6):
                out.write(json.dumps({
                    "title": title, "text": ptext,
                    "source": "wikipedia", "views": views
                }, ensure_ascii=False) + '\n')
                passages += 1
            matched += 1

            if matched % 2000 == 0:
                el = time.time() - start
                scanned = i - resume_from
                rate = scanned / el if el else 0
                remaining = total - i
                eta = remaining / rate / 3600 if rate else 0
                print(f"  WP: {i:,}/{total:,} ({i/total*100:.1f}%) "
                      f"| matched {matched:,} | {passages:,} passages "
                      f"| {rate:.0f}/s | ETA {eta:.1f}h", flush=True)

        except Exception:
            continue

        if i - last_checkpoint >= WIKI_CHECKPOINT_EVERY:
            out.flush()
            progress["wikipedia_resume_from"] = i + 1
            progress["wikipedia_matched"]     = matched
            progress["wikipedia_passages"]    = passages
            save_progress(progress)
            last_checkpoint = i

        if not probe_done and (i - resume_from) >= 500000:
            el = time.time() - start
            rate = (i - resume_from) / el if el else 0
            eta = (total - i) / rate / 3600 if rate else 0
            print(f"\n  *** SPEED PROBE: {rate:.0f} entries/sec ***", flush=True)
            print(f"  *** Projected Wikipedia time: {eta:.1f} hours ***\n",
                  flush=True)
            probe_done = True

    new_passages = passages - progress["wikipedia_passages"]
    elapsed = time.time() - start
    print(f"  DONE WIKIPEDIA: matched {matched:,} | "
          f"{passages:,} passages | {elapsed/60:.1f} min", flush=True)
    return new_passages


def extract_small_source(fname, tag, min_chars, max_passages,
                         max_articles, out):
    from libzim.reader import Archive

    path = os.path.join(ZIM_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP {fname} — not found.", flush=True)
        return 0

    archive   = Archive(path)
    total     = archive.all_entry_count
    get_entry = getattr(archive, 'get_entry_by_id',
                getattr(archive, '_get_entry_by_id', None))

    cap_str = "all" if max_articles is None else f"{max_articles:,}"
    print(f"\n=== {tag.upper()} ({fname}) ===", flush=True)
    print(f"  Entries: {total:,}  |  cap: {cap_str}", flush=True)

    passages = 0
    matched  = 0
    start    = time.time()

    for i in range(total):
        try:
            entry = get_entry(i)
            if entry.is_redirect: continue
            title = str(entry.title).strip()
            if not title or looks_like_junk(title): continue
            seg = str(entry.path).split('/')[-1]
            if '.' in seg and not seg.endswith('.html'): continue
            raw  = bytes(entry.get_item().content).decode('utf-8', errors='ignore')
            text = clean_html(raw)
            if len(text) < min_chars: continue

            for ptext in chunk_overlapping(text, max_passages):
                out.write(json.dumps({
                    "title": title, "text": ptext,
                    "source": tag, "views": 0
                }, ensure_ascii=False) + '\n')
                passages += 1
            matched += 1

            if matched % 3000 == 0:
                el = time.time() - start
                rate = matched / el if el else 0
                print(f"  {tag}: {matched:,} articles | "
                      f"{passages:,} passages | {rate:.0f}/s", flush=True)

            if max_articles is not None and matched >= max_articles:
                print(f"  Reached cap for {tag} — stopping.", flush=True)
                break
        except Exception:
            continue

    elapsed = time.time() - start
    print(f"  DONE {tag}: {matched:,} articles | "
          f"{passages:,} passages | {elapsed/60:.1f} min", flush=True)
    return passages


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("Loading title set...", flush=True)
    title_set = load_title_set(TITLES_FILE)

    progress = load_progress()
    completed = set(progress["completed_sources"])
    print(f"Completed sources: {completed if completed else 'none'}", flush=True)
    if "wikipedia" not in completed and progress["wikipedia_resume_from"] > 0:
        print(f"Wikipedia partial: resumes from entry "
              f"{progress['wikipedia_resume_from']:,}", flush=True)

    mode = 'a' if (completed or progress["wikipedia_resume_from"] > 0) else 'w'

    with open(LOCAL_PASSAGES, mode, encoding='utf-8', buffering=1) as local_f, \
         open(PASSAGES_FILE,  mode, encoding='utf-8', buffering=1) as ssd_f:

        class DualWriter:
            def write(self, s):
                local_f.write(s); ssd_f.write(s)
            def flush(self):
                local_f.flush(); ssd_f.flush()

        out = DualWriter()

        for fname, tag, use_filter, min_c, max_p, max_a in SOURCES:
            if tag in completed:
                print(f"\n  Skipping {tag} (already done).", flush=True)
                continue

            if tag == "wikipedia":
                n = extract_wikipedia(title_set, out, progress)
            else:
                n = extract_small_source(fname, tag, min_c, max_p, max_a, out)

            progress["total_passages"] += n
            progress["completed_sources"].append(tag)
            if tag == "wikipedia":
                progress["wikipedia_resume_from"] = 0
            save_progress(progress)
            out.flush()
            os.fsync(local_f.fileno())
            print(f"  Progress saved. Total so far: "
                  f"{progress['total_passages']:,} passages", flush=True)

    print(f"\n{'='*55}", flush=True)
    print(f"EXTRACTION COMPLETE", flush=True)
    print(f"Total passages: {progress['total_passages']:,}", flush=True)
    print(f"  SSD:   {PASSAGES_FILE}", flush=True)
    print(f"  Local: {LOCAL_PASSAGES}", flush=True)
    print(f"{'='*55}", flush=True)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == '__main__':
    main()