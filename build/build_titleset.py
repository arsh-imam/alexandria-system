"""
Downloads Wikipedia pageview data for one day, aggregates view counts,
and saves the top 200k most-viewed English article titles.
Output: G:\local_ai\index\top_200k_titles.txt

This runs on the HP laptop (no GPU/heavy compute needed).
The title list is then used on the Nitro 5 for fast targeted extraction.
"""

import os
import sys
import gzip
import time
import requests
from collections import Counter

# ── Config ───────────────────────────────────────────────────────────────
LOCAL_DIR   = os.path.join(os.path.expanduser("~"), "lcatrs_pageviews")
OUTPUT_DIR  = r"G:\local_ai\index"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top_200k_titles.txt")
TOP_K       = 200000

# Use 5 recent days for stable ranking (smooths daily spikes)
DATES = ["2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26"]
BASE_URL = "https://dumps.wikimedia.org/other/pageviews/2026/2026-05"

# 4 snapshots per day × 5 days = 20 files (manageable download)
HOURS = ["000000", "060000", "120000", "180000"]


def download_file(url, dest):
    """Download with retry and resume support."""
    for attempt in range(3):
        try:
            print(f"  Downloading: {os.path.basename(dest)}...",
                  end="", flush=True)
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 404:
                print(f" NOT FOUND (404)", flush=True)
                return False
            r.raise_for_status()
            size = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    size += len(chunk)
            print(f" OK ({size/1024/1024:.1f} MB)", flush=True)
            return True
        except Exception as e:
            print(f" RETRY ({e})", flush=True)
            time.sleep(2)
    print(f"  FAILED after 3 attempts.", flush=True)
    return False


def process_file(filepath, counter):
    """Parse one pageview gz file, add English Wikipedia counts."""
    added = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 3:
                    continue
                project = parts[0]
                # English Wikipedia articles only
                if project not in ('en', 'en.wikipedia'):
                    continue
                title = parts[1]
                try:
                    views = int(parts[2])
                except ValueError:
                    continue
                # Skip special pages
                if title.startswith(('Special:', 'Talk:', 'User:',
                    'Wikipedia:', 'File:', 'Template:', 'Help:',
                    'Category:', 'Portal:', 'Module:', 'Draft:',
                    'MediaWiki:', 'TimedText:', 'Main_Page')):
                    continue
                # Convert URL-encoded title to readable form
                title_clean = title.replace('_', ' ')
                counter[title_clean] += views
                added += 1
    except Exception as e:
        print(f"  Error processing {filepath}: {e}", flush=True)
    return added


def find_working_date():
    """Try the target date, fall back to earlier dates if not available."""
    date_str = TARGET_DATE.replace("-", "")
    test_url = f"{BASE_URL}/pageviews-{date_str}-{HOURS[0]}.gz"

    print(f"Checking availability at {TARGET_DATE}...", flush=True)
    try:
        r = requests.head(test_url, timeout=30)
        if r.status_code == 200:
            print(f"  Data available for {TARGET_DATE}.", flush=True)
            return date_str
    except:
        pass

    # Try earlier dates
    for day_offset in range(1, 15):
        day = 26 - day_offset
        if day < 1:
            month_str = "04"
            day = 30 + day  # wrap to April
            alt_base = f"https://dumps.wikimedia.org/other/pageviews/2026/2026-{month_str}"
        else:
            month_str = "05"
            alt_base = BASE_URL
        date_str = f"2026{month_str}{day:02d}"
        test_url = f"{alt_base}/pageviews-{date_str}-{HOURS[0]}.gz"
        try:
            r = requests.head(test_url, timeout=30)
            if r.status_code == 200:
                print(f"  Using fallback date: 2026-{month_str}-{day:02d}",
                      flush=True)
                return date_str
        except:
            continue

    return None


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Download hourly files across multiple days
    print("=" * 55, flush=True)
    print(f"STEP 1: Downloading pageview data for {len(DATES)} days...",
          flush=True)
    downloaded = []
    for date in DATES:
        date_str = date.replace("-", "")
        month = date_str[4:6]
        base = f"https://dumps.wikimedia.org/other/pageviews/2026/2026-{month}"
        for hour in HOURS:
            fname = f"pageviews-{date_str}-{hour}.gz"
            url   = f"{base}/{fname}"
            dest  = os.path.join(LOCAL_DIR, fname)

            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                print(f"  {fname} already downloaded, skipping.", flush=True)
                downloaded.append(dest)
                continue

            if download_file(url, dest):
                downloaded.append(dest)

    if len(downloaded) < 5:
        print(f"\nERROR: Only {len(downloaded)} files downloaded. "
              f"Need at least 5.", flush=True)
        sys.exit(1)
    print(f"\nDownloaded {len(downloaded)} files across {len(DATES)} days.",
          flush=True)

    # Step 2: Process and aggregate
    print(f"\n{'='*55}", flush=True)
    print("STEP 2: Processing pageview data...", flush=True)
    counter = Counter()
    for filepath in downloaded:
        print(f"  Processing {os.path.basename(filepath)}...",
              end="", flush=True)
        n = process_file(filepath, counter)
        print(f" {n:,} English entries", flush=True)

    total_articles = len(counter)
    print(f"\nTotal unique English articles found: {total_articles:,}",
          flush=True)

    if total_articles < TOP_K:
        print(f"WARNING: Only {total_articles:,} articles found, "
              f"less than target {TOP_K:,}.", flush=True)

    # Step 3: Filter out obvious junk
    print(f"\n{'='*55}", flush=True)
    print("STEP 3: Filtering junk entries...", flush=True)
    junk_prefixes = ('Special:', 'Talk:', 'User:', 'Wikipedia:', 'File:',
                     'Template:', 'Help:', 'Category:', 'Portal:',
                     'Module:', 'Draft:', 'MediaWiki:', 'TimedText:')
    filtered = Counter()
    for title, views in counter.items():
        if len(title) < 2:
            continue
        if title.startswith(junk_prefixes):
            continue
        if title in ('Main_Page', 'Main Page', '-', '–', '—'):
            continue
        # Skip titles that are just numbers or single characters
        if title.strip().isdigit():
            continue
        filtered[title] = views

    print(f"  Before filter: {total_articles:,}", flush=True)
    print(f"  After filter:  {len(filtered):,}", flush=True)

    # Step 4: Rank and save
    print(f"\n{'='*55}", flush=True)
    actual_k = min(TOP_K, len(filtered))
    print(f"STEP 4: Selecting top {actual_k:,} articles by views...",
          flush=True)
    top = filtered.most_common(actual_k)

    print(f"\n  Top 30 most viewed articles (5-day aggregate):", flush=True)
    for i, (title, views) in enumerate(top[:30], 1):
        print(f"    {i:3d}. {title} ({views:,} views)", flush=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for title, views in top:
            f.write(f"{views}\t{title}\n")
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nSaved: {OUTPUT_FILE}", flush=True)
    print(f"  {actual_k:,} titles, {size_kb:.0f} KB", flush=True)

    local_backup = os.path.join(LOCAL_DIR, "top_200k_titles.txt")
    with open(local_backup, 'w', encoding='utf-8') as f:
        for title, views in top:
            f.write(f"{views}\t{title}\n")
    print(f"  Backup: {local_backup}", flush=True)

    print(f"\nCleaning up downloaded files...", flush=True)
    for f_path in downloaded:
        try:
            os.remove(f_path)
        except:
            pass

    print(f"\n{'='*55}", flush=True)
    print("DONE. Title set ready for the Nitro 5 build.", flush=True)
    print(f"  {actual_k:,} most-viewed articles across {len(DATES)} days.",
          flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == '__main__':
    main()