"""
Builds the Annoy search index from pre-computed vectors.
Run once after GPU embedding. Takes ~15-30 min on Pi 5.
"""

import os
import time
import numpy as np
from annoy import AnnoyIndex

VECTORS   = "/media/pi/KINGSTON/local_ai/index/vectors.npy"
ANNOY_OUT = "/media/pi/KINGSTON/local_ai/index/passages.ann"
DIMS      = 384
TREES     = 50


def main():
    print("Loading vectors...", flush=True)
    vecs  = np.load(VECTORS, mmap_mode='r')
    total = vecs.shape[0]
    print(f"Shape: {vecs.shape}  ({total:,} passages)", flush=True)

    print(f"\nBuilding Annoy index ({TREES} trees)...", flush=True)
    print("This takes 15-30 minutes. Safe to leave running.\n", flush=True)
    t     = time.time()
    index = AnnoyIndex(DIMS, 'angular')

    for i in range(total):
        index.add_item(i, vecs[i].tolist())
        if i % 200000 == 0 and i > 0:
            elapsed = time.time() - t
            rate    = i / elapsed if elapsed else 0
            eta     = (total - i) / rate / 60 if rate else 0
            print(f"  {i:,}/{total:,} added | "
                  f"{rate:.0f}/sec | ETA {eta:.0f} min", flush=True)

    print("\nBuilding trees...", flush=True)
    index.build(TREES)
    index.save(ANNOY_OUT)

    elapsed = time.time() - t
    size_mb = os.path.getsize(ANNOY_OUT) / 1024 / 1024
    print(f"\nDone in {elapsed/60:.1f} min.", flush=True)
    print(f"Index: {ANNOY_OUT} ({size_mb:.0f} MB)", flush=True)
    print(f"Total passages indexed: {total:,}", flush=True)
    print("\nNext: run the main app — python3 main.py", flush=True)


if __name__ == '__main__':
    main()
