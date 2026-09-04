"""
GPU embedding on Nitro 5 (GTX 1650).
Uses float32 ONNX model for full GPU acceleration.
Chunk-saves to local disk every 100k passages — crash safe and resumable.
"""

import os, json, time, shutil
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ── Fix DLL paths (must happen before onnxruntime loads) ─────────────────
dll_base = r"F:\nitro_venv\Lib\site-packages\nvidia"
for sub in ["cublas","cudnn","cuda_runtime","cufft","nvjitlink"]:
    p = os.path.join(dll_base, sub, "bin")
    if os.path.exists(p) and p not in os.environ.get("PATH",""):
        os.environ["PATH"] = p + ";" + os.environ["PATH"]

# ── Paths ─────────────────────────────────────────────────────────────────
SSD_PASSAGES = r"F:\local_ai\index\passages.jsonl"
SSD_VECTORS  = r"F:\local_ai\index\vectors.npy"
LOCAL_DIR    = os.path.join(os.path.expanduser("~"), "lcatrs_embed")

MODEL_ONNX   = r"C:\Users\compu\AppData\Local\Temp\hf_models\models--BAAI--bge-small-en-v1.5\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a\onnx\model.onnx"
TOKENIZER    = r"C:\Users\compu\AppData\Local\Temp\hf_models\models--BAAI--bge-small-en-v1.5\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a\tokenizer.json"

DIMS       = 384
BATCH_SIZE = 32
CHUNK_SIZE = 100000


# ── Embedder ──────────────────────────────────────────────────────────────

class GPUEmbedder:
    def __init__(self):
        self.tok = Tokenizer.from_file(TOKENIZER)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self.tok.enable_truncation(max_length=128)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            MODEL_ONNX, so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        active = self.sess.get_providers()
        if "CUDAExecutionProvider" in active:
            print(">>> GPU (CUDA) ACTIVE <<<", flush=True)
        else:
            print(">>> WARNING: CPU only — check DLL paths <<<", flush=True)

    def embed(self, texts):
        enc   = self.tok.encode_batch(texts)
        ids   = np.array([e.ids for e in enc], dtype=np.int64)
        mask  = np.array([e.attention_mask for e in enc], dtype=np.int64)
        types = np.zeros_like(ids)
        out   = self.sess.run(None, {
            "input_ids": ids, "attention_mask": mask, "token_type_ids": types})
        embs  = out[0].mean(axis=1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-9)).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Count passages
    print("Counting passages...", flush=True)
    total = sum(1 for _ in open(SSD_PASSAGES, 'r', encoding='utf-8'))
    print(f"Total: {total:,} passages.", flush=True)

    # Check resume state
    existing = sorted([f for f in os.listdir(LOCAL_DIR)
                       if f.startswith("chunk_") and f.endswith(".npy")])
    start_from = len(existing) * CHUNK_SIZE
    if start_from > 0:
        print(f"Resuming from {start_from:,} "
              f"({len(existing)} chunks already saved).", flush=True)

    # Load all texts (SSD read-only)
    print("Loading passage texts...", flush=True)
    texts = []
    with open(SSD_PASSAGES, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"Loaded {len(texts):,} texts.", flush=True)

    # Load embedder
    print("Loading GPU embedder...", flush=True)
    emb = GPUEmbedder()
    emb.embed(["warmup"])

    # Embed
    print(f"\nEmbedding from {start_from:,} to {total:,}...\n", flush=True)
    start_time = time.time()
    chunk_vecs = []
    chunk_num  = len(existing)
    done       = start_from

    for b in range(start_from, total, BATCH_SIZE):
        batch = texts[b : b + BATCH_SIZE]
        vecs  = emb.embed(batch)
        chunk_vecs.append(vecs)
        done  = b + len(vecs)

        # Save chunk every CHUNK_SIZE passages
        if done % CHUNK_SIZE < BATCH_SIZE and done > start_from:
            arr  = np.vstack(chunk_vecs)
            path = os.path.join(LOCAL_DIR, f"chunk_{chunk_num:04d}.npy")
            np.save(path, arr)
            chunk_num += 1
            chunk_vecs = []

            elapsed  = time.time() - start_time
            embedded = done - start_from
            rate     = embedded / elapsed if elapsed else 0
            eta      = (total - done) / rate / 60 if rate else 0
            print(f"  [{done/total*100:5.1f}%] {done:,}/{total:,} | "
                  f"{rate:.0f}/sec | ETA {eta:.0f} min  "
                  f"[chunk {chunk_num} saved to C:]", flush=True)

        # Progress every 10k
        elif done % 10000 < BATCH_SIZE and done > start_from:
            elapsed  = time.time() - start_time
            embedded = done - start_from
            rate     = embedded / elapsed if elapsed else 0
            eta      = (total - done) / rate / 60 if rate else 0
            print(f"  [{done/total*100:5.1f}%] {done:,}/{total:,} | "
                  f"{rate:.0f}/sec | ETA {eta:.0f} min", flush=True)

    # Save final chunk
    if chunk_vecs:
        arr  = np.vstack(chunk_vecs)
        path = os.path.join(LOCAL_DIR, f"chunk_{chunk_num:04d}.npy")
        np.save(path, arr)
        print(f"  Final chunk saved.", flush=True)

    # Concatenate all chunks
    print("\nConcatenating chunks...", flush=True)
    all_chunks = sorted([f for f in os.listdir(LOCAL_DIR)
                         if f.startswith("chunk_") and f.endswith(".npy")])
    full = np.vstack([np.load(os.path.join(LOCAL_DIR, f))
                      for f in all_chunks])
    print(f"Final shape: {full.shape}", flush=True)

    # Save locally first
    local_vec = os.path.join(LOCAL_DIR, "vectors.npy")
    np.save(local_vec, full)
    local_mb = os.path.getsize(local_vec) / 1024 / 1024
    print(f"Saved locally: {local_mb:.0f} MB", flush=True)

    # Copy to SSD
    print(f"Copying to SSD...", flush=True)
    try:
        shutil.copy2(local_vec, SSD_VECTORS)
        ssd_mb = os.path.getsize(SSD_VECTORS) / 1024 / 1024
        print(f"Copied to SSD: {ssd_mb:.0f} MB", flush=True)
    except Exception as e:
        print(f"SSD copy failed: {e}", flush=True)
        print(f"Vectors safe at: {local_vec}", flush=True)
        print(f"Manually copy to: {SSD_VECTORS}", flush=True)
        return

    elapsed = time.time() - start_time
    rate    = (total - start_from) / elapsed if elapsed else 0
    print(f"\n{'='*55}", flush=True)
    print(f"DONE. {total:,} passages in {elapsed/60:.1f} min "
          f"({rate:.0f}/sec).", flush=True)
    print(f"Passages: {SSD_PASSAGES}", flush=True)
    print(f"Vectors:  {SSD_VECTORS}", flush=True)
    print(f"\nNext: move SSD to Pi, run build_annoy.py", flush=True)
    print(f"{'='*55}", flush=True)

    # Cleanup local temp files
    for f in all_chunks:
        os.remove(os.path.join(LOCAL_DIR, f))
    if os.path.exists(local_vec):
        os.remove(local_vec)
    print("Cleaned up local temp files.", flush=True)


if __name__ == '__main__':
    main()