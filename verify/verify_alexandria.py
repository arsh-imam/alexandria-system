#!/usr/bin/env python3
"""
ALEXANDRIA verification suite.

Nine gates, G0-G8. Each declares its own prerequisites and SKIPs cleanly when they
are absent, so the same script is useful to a reviewer with nothing but a
clone and to one with the full 96.7 GB deployment.

    python3 verify_alexandria.py                    # run every applicable gate
    python3 verify_alexandria.py --gates G2 G3      # run specific gates
    python3 verify_alexandria.py --list
    python3 verify_alexandria.py --rebuild-fixtures # on the Pi, regenerate G3 data
    python3 verify_alexandria.py --deep             # G6 re-hashes 96.7 GB

Exit code 0 only if no gate FAILED. SKIP is not a failure; the summary states
exactly which claims were and were not checked.

Gates
  G0  repo integrity      code files match code_hashes.tsv
  G1  environment         Python and pinned package versions
  G2  embedding alignment self_cos >= 0.98  (the central engineering claim)
  G3  chunking parameters 160-word windows, 40-word overlap, per-source caps
  G4  index integrity     the four-way row-index join
  G5  llama.cpp build     aarch64 CPU_REPACK kernels active
  G6  corpus integrity    41 archives match sha256 and ZIM UUID
  G7  retrieval golden    deterministic retrieval on a fixed query set
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


_UP = os.path.dirname(HERE)
CODE = os.environ.get("ALEX_CODE") or _first_dir(
    os.path.join(_UP, "src"), os.path.join(HERE, "src"), HERE) or HERE
MANIFESTS = os.environ.get("ALEX_MANIFESTS") or _first_dir(
    os.path.join(_UP, "manifests"),
    os.path.join(HERE, "repro_bundle", "manifests"),
    os.path.join(_UP, "repro_bundle", "manifests"),
) or os.path.join(HERE, "repro_bundle", "manifests")
FIXTURES = os.environ.get("ALEX_FIXTURES") or _first_dir(
    os.path.join(HERE, "fixtures"),
    os.path.join(HERE, "repro_bundle", "fixtures"),
    os.path.join(_UP, "repro_bundle", "fixtures"),
) or os.path.join(HERE, "repro_bundle", "fixtures")
BUNDLE = os.environ.get("ALEX_BUNDLE") or os.path.dirname(MANIFESTS)
SSD = os.environ.get("ALEX_ROOT", "/media/pi/KINGSTON/local_ai")

ALIGN_THRESHOLD = 0.98
WORDS_PER_PASSAGE = 160
OVERLAP_WORDS = 40
SOURCE_CAPS = {"wikipedia": 6, "wikibooks": 6, "ifixit": 5, "wikivoyage": 5,
               "wikiquote": 2, "wikinews": 2, "wiktionary": 1}

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_gates, _results = {}, []


def gate(gid, title, needs):
    def deco(fn):
        _gates[gid] = (title, needs, fn)
        return fn
    return deco


class Skip(Exception):
    pass


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _quiet_ort():
    """Silence ONNX Runtime's GPU device-discovery warnings on headless ARM."""
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
    except Exception:
        pass


def read_tsv(path):
    if not os.path.exists(path):
        raise Skip(f"missing {os.path.relpath(path, BUNDLE)}")
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, ln.rstrip("\n").split("\t")))
                for ln in f if ln.strip()]


# ---------------------------------------------------------------- G0

@gate("G0", "repo integrity", "clone only")
def g0():
    rows = read_tsv(os.path.join(MANIFESTS, "code_hashes.tsv"))
    ok, bad, absent = 0, [], []
    for r in rows:
        p = os.path.join(CODE, r["module"])
        if not os.path.exists(p):
            absent.append(r["module"])
            continue
        if r["sha256"] in ("MISSING", ""):
            continue
        if sha256(p) == r["sha256"]:
            ok += 1
        else:
            bad.append(r["module"])
    if bad:
        return FAIL, f"{len(bad)} modified: {bad}"
    if absent:
        return FAIL, f"{len(absent)} missing: {absent}"
    return PASS, f"{ok} modules byte-identical to manifest"


# ---------------------------------------------------------------- G1

@gate("G1", "environment", "clone only")
def g1():
    lock = next((q for q in (os.path.join(MANIFESTS, "requirements-frozen.txt"),
                         os.path.join(MANIFESTS, "env_development.txt"),
                         os.path.join(MANIFESTS, "requirements-pi.lock"))
             if os.path.exists(q)), os.path.join(MANIFESTS, "requirements-frozen.txt"))
    if not os.path.exists(lock):
        raise Skip("no requirements lock found in manifests/")
    pinned = {}
    for ln in open(lock, encoding="utf-8"):
        if "==" in ln:
            k, v = ln.strip().split("==", 1)
            pinned[k.lower().replace("-", "_")] = v

    critical = ["tokenizers", "numpy", "onnxruntime", "annoy", "libzim",
                "llama_cpp_python"]
    from importlib.metadata import version, PackageNotFoundError
    issues, checked = [], 0
    for name in critical:
        want = pinned.get(name)
        if not want:
            issues.append(f"{name}: not in lock")
            continue
        for cand in (name, name.replace("_", "-")):
            try:
                have = version(cand)
                break
            except PackageNotFoundError:
                have = None
        if have is None:
            issues.append(f"{name}: not installed (need {want})")
        elif have != want:
            issues.append(f"{name}: have {have}, need {want}")
        else:
            checked += 1
    if issues:
        return FAIL, "; ".join(issues)
    return PASS, f"{checked}/{len(critical)} critical pins match, "\
                 f"{len(pinned)} total in lock"


# ---------------------------------------------------------------- G2

@gate("G2", "embedding alignment (self_cos >= 0.98)", "clone + embedder model")
def g2():
    import numpy as np
    fx_v = os.path.join(FIXTURES, "alignment_vectors.npy")
    fx_p = os.path.join(FIXTURES, "alignment_passages.jsonl")
    if not (os.path.exists(fx_v) and os.path.exists(fx_p)):
        raise Skip("alignment fixture not present")

    embed_dir = os.environ.get("ALEX_EMBED_DIR",
                               os.path.join(SSD, "models/embed"))
    onnx = os.path.join(embed_dir, "model.onnx")
    tokj = os.path.join(embed_dir, "tokenizer.json")
    if not (os.path.exists(onnx) and os.path.exists(tokj)):
        # Not installed: fetch the pinned revision into a cache beside this
        # file so the gate runs from a bare clone. 133 MB, verified by hash.
        REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        BASE = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/" + REV + "/"
        WANT = {"model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
                "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"}
        embed_dir = os.path.join(HERE, ".embedder-cache")
        os.makedirs(embed_dir, exist_ok=True)
        onnx = os.path.join(embed_dir, "model.onnx")
        tokj = os.path.join(embed_dir, "tokenizer.json")
        import urllib.request, urllib.error
        for fn, want in WANT.items():
            dst = os.path.join(embed_dir, fn)
            if os.path.exists(dst) and sha256(dst) == want:
                continue
            url = BASE + ("onnx/" if fn.endswith(".onnx") else "") + fn
            print("      fetching %s (pinned revision)" % fn, flush=True)
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:
                raise Skip("embedder absent and download failed (%s). "
                           "Install the system, or set ALEX_EMBED_DIR." % type(e).__name__)
            if sha256(dst) != want:
                os.remove(dst)
                raise Skip("downloaded embedder failed its hash check")

    _quiet_ort()
    sys.path.insert(0, CODE)
    try:
        import embedder as E
    except Exception as e:
        raise Skip(f"cannot import embedder.py: {e}")
    # redirect where the files live; the pooling code under test is untouched
    E.MODEL_ONNX, E.TOKENIZER_JSON = onnx, tokj
    emb = E.AlignedEmbedder()

    stored = np.load(fx_v)
    texts = [json.loads(l)["text"] for l in open(fx_p, encoding="utf-8")]
    if len(texts) != stored.shape[0]:
        return FAIL, f"fixture mismatch: {len(texts)} texts vs {stored.shape}"

    cos, qs, B = [], [], 64
    for i in range(0, len(texts), B):
        q = emb.embed(texts[i:i + B])
        qs.append(q)
        cos.extend((q * stored[i:i + B]).sum(axis=1).tolist())
    qmat = np.vstack(qs)
    cos = np.array(cos)
    worst, mean = float(cos.min()), float(cos.mean())
    below = int((cos < ALIGN_THRESHOLD).sum())
    tiers = {t: int((cos < t).sum()) for t in (0.999, 0.9999)}

    # Negative control. A near-1.0 result is only meaningful if the same
    # measurement can produce a low one, so score every query vector against
    # a different passage's stored vector and report the floor.
    n = len(cos)
    idx = np.arange(n)
    perm = (idx + 1 + np.random.default_rng(0).integers(0, n - 1, n)) % n
    rand = (qmat * stored[perm]).sum(axis=1)
    rmean, rmax = float(rand.mean()), float(rand.max())

    detail = (f"n={n} worst={worst:.6f} mean={mean:.6f} below_0.98={below} "
              f"(<0.999: {tiers[0.999]}, <0.9999: {tiers[0.9999]}) | "
              f"random-pair control mean={rmean:.4f} max={rmax:.4f}")

    mm = _mismatch_control(texts[:128], stored[:128])
    if mm:
        detail += " | " + mm

    if worst < ALIGN_THRESHOLD:
        return FAIL, detail + "  -> query path does NOT match the index"
    if rmax >= ALIGN_THRESHOLD:
        return FAIL, detail + "  -> control failed: gate cannot discriminate"
    return PASS, detail


def _mismatch_control(texts, stored):
    """Re-run the historical broken path (bare fastembed: quantised weights,
    CLS pooling) against the same vectors. It must FAIL the threshold, which
    is what proves this gate detects the defect it was built for."""
    try:
        import numpy as np
        from fastembed import TextEmbedding
        fe = TextEmbedding("BAAI/bge-small-en-v1.5")
        v = np.asarray(list(fe.embed(list(texts))), dtype=np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        c = (v * stored).sum(axis=1)
        lo, mu = float(c.min()), float(c.mean())
        verdict = ("correctly FAILS 0.98" if lo < ALIGN_THRESHOLD
                   else "UNEXPECTEDLY PASSES - investigate")
        return (f"mismatch control (bare fastembed, n={len(texts)}): "
                f"worst={lo:.4f} mean={mu:.4f} -> {verdict}")
    except Exception as e:
        return f"mismatch control unavailable ({type(e).__name__})"


# ---------------------------------------------------------------- G3

@gate("G3", "chunking parameters", "clone only")
def g3():
    fx = os.path.join(FIXTURES, "chunking_cases.json")
    if not os.path.exists(fx):
        raise Skip("chunking fixture not present (--rebuild-fixtures)")
    data = json.load(open(fx, encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases or "passages" not in cases[0]:
        raise Skip("fixture predates the overlap check "
                   "(regenerate with --rebuild-fixtures)")

    step = WORDS_PER_PASSAGE - OVERLAP_WORDS
    bad, n_pass, n_art = [], 0, 0
    for c in cases:
        texts = c["passages"]
        src = c.get("source", "wikipedia")
        cap = SOURCE_CAPS.get(src, 6)
        n_art += 1
        if len(texts) > cap:
            bad.append(f"{c['title']}: {len(texts)} passages > cap {cap}")
        for i, t in enumerate(texts):
            w = t.split()
            n_pass += 1
            if len(w) > WORDS_PER_PASSAGE:
                bad.append(f"{c['title']}[{i}]: {len(w)} words > "
                           f"{WORDS_PER_PASSAGE}")
            if len(w) <= OVERLAP_WORDS and i != len(texts) - 1:
                bad.append(f"{c['title']}[{i}]: {len(w)} words <= "
                           f"{OVERLAP_WORDS} but not final")
            if i + 1 < len(texts):
                nxt = texts[i + 1].split()
                tail, head = w[step:], nxt[:OVERLAP_WORDS]
                if len(w) == WORDS_PER_PASSAGE and tail != head[:len(tail)]:
                    bad.append(f"{c['title']}[{i}->{i+1}]: "
                               f"{OVERLAP_WORDS}-word overlap broken")
    if bad:
        return FAIL, f"{len(bad)} violations, first: {bad[0]}"
    return PASS, (f"{n_art} articles / {n_pass} passages: "
                  f"{WORDS_PER_PASSAGE}w windows, {OVERLAP_WORDS}w overlap, "
                  f"caps respected")


# ---------------------------------------------------------------- G4

@gate("G4", "index cardinality and ID-layout integrity", "index artifacts")
def g4():
    import sqlite3
    idx = os.path.join(SSD, "index")
    need = ["passages.db", "vectors.npy", "passages.ann", "passages.jsonl"]
    absent = [n for n in need if not os.path.exists(os.path.join(idx, n))]
    if absent:
        raise Skip(f"index artifacts absent: {absent}")

    import numpy as np
    from annoy import AnnoyIndex
    db = sqlite3.connect(os.path.join(idx, "passages.db"))
    n_db = db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    lo, hi = db.execute("SELECT MIN(id), MAX(id) FROM passages").fetchone()
    n_fts = db.execute("SELECT COUNT(*) FROM passages_fts").fetchone()[0]
    per = dict(db.execute("SELECT source, COUNT(*) FROM passages "
                          "GROUP BY source").fetchall())
    db.close()

    v = np.load(os.path.join(idx, "vectors.npy"), mmap_mode="r")
    ann = AnnoyIndex(384, "angular")
    ann.load(os.path.join(idx, "passages.ann"), prefault=False)
    n_jsonl = sum(1 for _ in open(os.path.join(idx, "passages.jsonl"),
                                  encoding="utf-8"))

    expect_npy = int(v.shape[0]) * int(v.shape[1]) * 4 + 128
    actual_npy = os.path.getsize(os.path.join(idx, "vectors.npy"))

    probs = []
    if not (n_db == n_fts == n_jsonl == v.shape[0] == ann.get_n_items()):
        probs.append(f"counts differ: db={n_db} fts={n_fts} jsonl={n_jsonl} "
                     f"npy={v.shape[0]} ann={ann.get_n_items()}")
    if (lo, hi) != (0, n_db - 1):
        probs.append(f"id range {lo}..{hi}, expected 0..{n_db-1}")
    if expect_npy != actual_npy:
        probs.append(f"vectors.npy size {actual_npy} != {expect_npy}")
    if str(v.dtype) != "float32":
        probs.append(f"vectors dtype {v.dtype}, expected float32")
    for s, cap in SOURCE_CAPS.items():
        if s not in per:
            probs.append(f"source '{s}' absent from index")
    if probs:
        return FAIL, "; ".join(probs)
    return PASS, (f"all four = {n_db:,}; ids 0..{n_db-1}; "
                  f"{len(per)} sources; npy size exact")


# ---------------------------------------------------------------- G5

@gate("G5", "llama.cpp aarch64 repack kernels", "GGUF, aarch64 only")
def g5():
    import platform as _pl
    if _pl.machine() != "aarch64":
        raise Skip("CPU_REPACK reference applies only to aarch64; host is %s"
                   % _pl.machine())
    gguf = os.environ.get("ALEX_GGUF",
                          os.path.join(SSD, "models/Qwen3-1.7B-Q4_K_M.gguf"))
    if not os.path.exists(gguf):
        raise Skip(f"GGUF not found at {gguf} (set ALEX_GGUF)")
    probe = ("from llama_cpp import Llama\n"
             f"Llama(model_path={gguf!r}, n_ctx=256, n_threads=4, "
             "verbose=True)\n")
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, timeout=900)
    blob = (r.stdout or "") + (r.stderr or "")
    if not blob.strip():
        raise Skip("no verbose output captured")
    kernels = sorted({t for t in blob.split()
                      if t.startswith(("q4_K_", "q6_K_")) and "x" in t})
    if "CPU_REPACK" not in blob:
        return FAIL, ("CPU_REPACK absent: this build is NOT using "
                      "aarch64-optimised kernels; throughput figures in the "
                      "paper are not comparable")
    return PASS, f"CPU_REPACK active, kernels {kernels}"


# ---------------------------------------------------------------- G6

@gate("G6", "corpus integrity (41 archives)", "full corpus")
def g6(deep=False):
    rows = read_tsv(os.path.join(MANIFESTS, "corpus_manifest.tsv"))
    zims = [r for r in rows if r.get("kind") == "zim"]
    if not zims:
        raise Skip("no ZIM rows in corpus manifest")

    present = [r for r in zims
               if os.path.exists(os.path.join(SSD, r["local_name"]))]
    if not present:
        raise Skip(f"no archives found under {SSD}")

    bad, checked = [], 0
    for r in present:
        p = os.path.join(SSD, r["local_name"])
        if str(os.path.getsize(p)) != r["size_bytes"]:
            bad.append(f"{r['local_name']}: size differs")
            continue
        try:
            from libzim.reader import Archive
            uid = str(Archive(p).uuid)
            if r["zim_uuid"] and uid != r["zim_uuid"]:
                bad.append(f"{r['local_name']}: UUID {uid} != {r['zim_uuid']}")
                continue
        except Exception as e:
            bad.append(f"{r['local_name']}: unreadable ({e})")
            continue
        if deep and sha256(p) != r["sha256"]:
            bad.append(f"{r['local_name']}: SHA256 MISMATCH")
            continue
        checked += 1

    mode = "size+uuid+sha256" if deep else "size+uuid"
    missing = len(zims) - len(present)
    if bad:
        return FAIL, f"{len(bad)} archives bad ({mode}), first: {bad[0]}"
    if missing:
        return FAIL, (f"{checked}/{len(zims)} verified ({mode}); "
                      f"{missing} archives ABSENT - deep tier incomplete")
    return PASS, f"{checked}/{len(zims)} archives verified ({mode})"


# ---------------------------------------------------------------- G7

@gate("G7", "retrieval golden set (determinism)", "full deployment")
def g7():
    golden = os.path.join(FIXTURES, "retrieval_golden.json")
    if not os.path.exists(golden):
        raise Skip("golden set absent (--rebuild-fixtures on a full system)")
    if not os.path.exists(os.path.join(SSD, "index/passages.ann")):
        raise Skip("index absent")
    # The deep tier opens only the archives in the registry, not all 41 on
    # disk. Derive the requirement rather than hard-coding a count.
    sys.path.insert(0, CODE)
    import glob as _g
    import archive_registry as _ar
    need = len(_ar.ARCHIVES)
    # ARCHIVES patterns are absolute to the module's own base path; rebase
    # them onto SSD so the check honours ALEX_ROOT.
    def _rebased(pat):
        i = pat.find("/local_ai/")
        return os.path.join(SSD, pat[i + len("/local_ai/"):]) if i >= 0 else pat
    have = sum(1 for spec in _ar.ARCHIVES.values()
               if _g.glob(_rebased(spec["pat"])))
    if have < need:
        raise Skip("%d of %d registered archives present; the golden set "
                   "exercises the deep tier and needs them all" % (have, need))

    _quiet_ort()
    sys.path.insert(0, CODE)
    import retriever as _R
    # main.py enables epistemic fusion at startup; RETRIEVAL_CONFIG defaults it
    # off. Verify the configuration the application actually runs.
    _R.RETRIEVAL_CONFIG["use_epistemic_fusion"] = True
    ref = json.load(open(golden, encoding="utf-8"))
    r = _R.Retriever()
    r.load()

    exact, neighbour_drift, corrupt, logit_drift = 0, [], [], []
    for case in ref["cases"]:
        got, conf, _ = r.search(case["query"])
        titles = [x["title"] for x in got]
        srcs = [x["source"] for x in got]
        shas = [hashlib.sha256(x["text"].encode()).hexdigest()[:16]
                for x in got]
        ref_shas = case.get("text_sha256")

        if titles != case["titles"] or srcs != case["sources"]:
            # a different article surfaced: the only thing approximate
            # nearest-neighbour search is allowed to change
            overlap = len(set(titles) & set(case["titles"]))
            neighbour_drift.append(f"{case['query'][:38]!r}: {overlap}/"
                                   f"{len(case['titles'])} titles overlap")
        elif ref_shas and shas != ref_shas:
            # same articles, different passage text: the row-index join or
            # the chunking has changed. Never a tolerable difference.
            corrupt.append(f"{case['query'][:38]!r}: same titles, "
                           f"different passage text")
        else:
            exact += 1
            if [x["logit"] for x in got] != case["logits"]:
                logit_drift.append(case["query"][:38])

    n = len(ref["cases"])
    # The .ann index is distributed and hash-pinned, so retrieval is
    # expected to reproduce exactly. No drift is tolerated.
    allowance = 0
    note = f"; {len(logit_drift)} with logit drift" if logit_drift else ""

    if corrupt:
        return FAIL, (f"{len(corrupt)} queries returned the same articles with "
                      f"DIFFERENT passage text - the row-index join or the "
                      f"chunking differs from the reference. First: "
                      f"{corrupt[0]}")
    if exact == n:
        return PASS, (f"{n}/{n} queries reproduce identical passages "
                      f"(title+source+text hash){note}")
    if len(neighbour_drift) <= allowance:
        return PASS, (f"{exact}/{n} identical; {len(neighbour_drift)} within "
                      f"the Annoy allowance of {allowance}: "
                      f"{neighbour_drift[0]}{note}")
    return FAIL, (f"{exact}/{n} identical; {len(neighbour_drift)} drifted, "
                  f"allowance {allowance}. First: {neighbour_drift[0]}")


@gate("G8", "generation determinism (byte-identical decode)", "GGUF only")
def g8():
    golden = os.path.join(FIXTURES, "generation_golden.json")
    gguf = os.environ.get("ALEX_GGUF",
                          os.path.join(SSD, "models/Qwen3-1.7B-Q4_K_M.gguf"))
    if not os.path.exists(golden):
        raise Skip("generation golden absent (--rebuild-fixtures)")
    if not os.path.exists(gguf):
        raise Skip(f"GGUF not found at {gguf} (set ALEX_GGUF)")

    ref = json.load(open(golden, encoding="utf-8"))
    import platform as _pl
    want_arch = ref.get("arch", "aarch64")
    have_arch = _pl.machine()
    if have_arch != want_arch:
        raise Skip("golden recorded on %s, host is %s. llama.cpp selects "
                   "different SIMD kernels per architecture, so decode is not "
                   "byte-identical across them (measured, not assumed)."
                   % (want_arch, have_arch))
    have_ver = _pkg_ver("llama_cpp_python")
    want_ver = ref.get("llama_cpp_python")
    notes = []
    if want_ver and have_ver != want_ver:
        notes.append(f"VERSION DRIFT: llama-cpp-python {have_ver} != {want_ver}")

    bad = []
    for c in ref["cases"]:
        got = _generate(gguf, c["prompt"], c["temperature"],
                        c["max_tokens"], ref["top_p"], ref["repeat_penalty"],
                        ref["n_ctx"], ref["n_threads"], ref["n_batch"])
        h = hashlib.sha256(got.encode()).hexdigest()
        if h != c["sha256"]:
            bad.append(f"temp {c['temperature']}: {h[:12]} != {c['sha256'][:12]}")

    detail = (f"{len(ref['cases'])} prompts x fresh load, "
              f"llama-cpp-python {have_ver}")
    if notes:
        detail += " | " + "; ".join(notes)
    if bad:
        return FAIL, (detail + " | decode differs: " + "; ".join(bad) +
                      " -> generation is NOT byte-reproducible on this build")
    return PASS, detail + " | all byte-identical to golden"


def _pkg_ver(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _generate(gguf, prompt, temp, n, top_p, rep, n_ctx, n_thr, n_bat):
    """Fresh load each call: a cached context would hide seed behaviour."""
    import llama_cpp
    llm = llama_cpp.Llama(model_path=gguf, n_ctx=n_ctx, n_threads=n_thr,
                          n_batch=n_bat, verbose=False)
    try:
        return llm.create_chat_completion(
            [{"role": "system",
              "content": "You are a helpful assistant. /no_think"},
             {"role": "user", "content": prompt}],
            max_tokens=n, temperature=temp, top_p=top_p,
            repeat_penalty=rep)["choices"][0]["message"]["content"]
    finally:
        del llm


# ---------------------------------------------------------- fixtures

def rebuild_fixtures():
    """Run on a full deployment. Regenerates G3 and G7 fixtures."""
    import sqlite3
    os.makedirs(FIXTURES, exist_ok=True)
    db = sqlite3.connect(os.path.join(SSD, "index/passages.db"))

    print("[G3] chunking fixture")
    cases = []
    for src in SOURCE_CAPS:
        titles = [r[0] for r in db.execute(
            "SELECT title FROM passages WHERE source=? "
            "GROUP BY title HAVING COUNT(*) >= 2 LIMIT 6", (src,)).fetchall()]
        for t in titles:
            texts = [r[0] for r in db.execute(
                "SELECT text FROM passages WHERE title=? AND source=? "
                "ORDER BY id", (t, src)).fetchall()]
            cases.append({"title": t, "source": src, "passages": texts})
    json.dump({"params": {"WORDS_PER_PASSAGE": WORDS_PER_PASSAGE,
                          "OVERLAP_WORDS": OVERLAP_WORDS,
                          "step": WORDS_PER_PASSAGE - OVERLAP_WORDS,
                          "caps": SOURCE_CAPS},
               "cases": cases},
              open(os.path.join(FIXTURES, "chunking_cases.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"     {len(cases)} articles across {len(SOURCE_CAPS)} sources")
    db.close()

    print("[G7] retrieval golden set (loads the full retriever)")
    queries = [
        "how do I fix a leaking kitchen tap",
        "what causes the northern lights",
        "does drinking coffee dehydrate you",
        "explain the Krebs cycle",
        "best time to visit Kyoto",
        "how to season a cast iron pan",
        "what is compound interest",
        "symptoms of vitamin D deficiency",
        "why did the Roman Republic fall",
        "how does a heat pump work",
        "what is the difference between RAM and ROM",
        "how do I keep basil alive indoors",
    ]
    _quiet_ort()
    sys.path.insert(0, CODE)
    import retriever as _R
    _R.RETRIEVAL_CONFIG["use_epistemic_fusion"] = True   # as main.py does
    r = _R.Retriever()
    r.load()
    cases = []
    for q in queries:
        # search() returns (results, confidence, search_time)
        res, conf, _ = r.search(q)
        cases.append({
            "query":      q,
            "titles":     [x["title"] for x in res],
            "sources":    [x["source"] for x in res],
            "logits":     [x["logit"] for x in res],
            "tiers":      [x["tier"] for x in res],
            "origins":    [x["origin"] for x in res],
            "confidence": conf,
            "text_sha256": [hashlib.sha256(x["text"].encode()).hexdigest()[:16]
                            for x in res],
        })
        print(f"     {q[:44]:<46} -> {len(res)} results  conf={conf}")
    json.dump({"k": 3, "generated": time.strftime("%Y-%m-%d"),
               "note": ("retrieval has no sampling; only Annoy's approximate "
                        "neighbours can vary, and only if the .ann was rebuilt"),
               "cases": cases},
              open(os.path.join(FIXTURES, "retrieval_golden.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)

    print("[G8] generation golden (3 fresh model loads)")
    import llama_cpp
    gguf = os.environ.get("ALEX_GGUF",
                          os.path.join(SSD, "models/Qwen3-1.7B-Q4_K_M.gguf"))
    PROMPT = ("Write an original four-line poem about a lighthouse keeper who "
              "has never once seen a ship. Do not rhyme.")
    params = {"top_p": 0.9, "repeat_penalty": 1.1, "n_ctx": 4096,
              "n_threads": 4, "n_batch": 512}
    cases = []
    for temp in (0.2, 0.4, 0.6):          # grounded / tentative / warm persona
        txt = _generate(gguf, PROMPT, temp, 160, params["top_p"],
                        params["repeat_penalty"], params["n_ctx"],
                        params["n_threads"], params["n_batch"])
        h = hashlib.sha256(txt.encode()).hexdigest()
        cases.append({"prompt": PROMPT, "temperature": temp,
                      "max_tokens": 160, "sha256": h, "text": txt})
        print(f"     temp {temp} -> {h[:12]}")
    import platform as _pl
    json.dump({"llama_cpp_python": _pkg_ver("llama_cpp_python"),
               "arch": _pl.machine(),
               "gguf": os.path.basename(gguf),
               "note": ("llama-cpp-python passes LLAMA_DEFAULT_SEED (0xffffffff) "
                        "through as a fixed seed rather than drawing randomly, "
                        "so decode is byte-reproducible at every deployed "
                        "temperature. Verified: explicit seeds DO change output, "
                        "the default does not. This is version-dependent."),
               **params, "cases": cases},
              open(os.path.join(FIXTURES, "generation_golden.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\nfixtures written to {FIXTURES}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="*", help="e.g. --gates G2 G3")
    ap.add_argument("--deep", action="store_true",
                    help="G6 re-hashes every archive (~96 GB)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rebuild-fixtures", action="store_true")
    args = ap.parse_args()

    if args.list:
        for gid in sorted(_gates):
            title, needs, _ = _gates[gid]
            print(f"  {gid}  {title:<44} needs: {needs}")
        return 0
    if args.rebuild_fixtures:
        rebuild_fixtures()
        return 0

    _mk = lambda p: "ok " if os.path.isdir(p) else "NOT FOUND"
    print("ALEXANDRIA verification")
    print(f"  code      {_mk(CODE)}  {CODE}")
    print(f"  manifests {_mk(MANIFESTS)}  {MANIFESTS}")
    print(f"  fixtures  {_mk(FIXTURES)}  {FIXTURES}")
    print(f"  corpus    {_mk(SSD)}  {SSD}")
    print()
    todo = sorted(args.gates) if args.gates else sorted(_gates)
    for gid in todo:
        if gid not in _gates:
            print(f"unknown gate {gid}")
            continue
        title, needs, fn = _gates[gid]
        t0 = time.time()
        try:
            status, detail = fn(args.deep) if gid == "G6" else fn()
        except Skip as e:
            status, detail = SKIP, str(e)
        except Exception as e:
            status, detail = FAIL, f"{type(e).__name__}: {e}"
        _results.append((gid, title, status, detail, time.time() - t0))
        print(f"{status}  {gid}  {title}\n      {detail}  "
              f"({time.time()-t0:.1f}s)\n")

    n_pass = sum(1 for r in _results if r[2] == PASS)
    n_fail = sum(1 for r in _results if r[2] == FAIL)
    n_skip = sum(1 for r in _results if r[2] == SKIP)
    print("=" * 72)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_skip:
        print("\nNOT CHECKED (prerequisites absent):")
        for gid, title, st, detail, _ in _results:
            if st == SKIP:
                print(f"  {gid}  {title}  -  {detail}")
        print("\nA skipped gate is not a passed gate. Full-system "
              "reproduction requires G4, G6 and G7.")
    try:
        os.makedirs(BUNDLE, exist_ok=True)
        with open(os.path.join(BUNDLE, "VERIFICATION_REPORT.tsv"), "w",
                  encoding="utf-8") as f:
            f.write("gate\ttitle\tstatus\tdetail\tseconds\n")
            for gid, title, st, detail, secs in _results:
                f.write(f"{gid}\t{title}\t{st}\t{detail}\t{secs:.1f}\n")
    except OSError as e:
        print(f"\n(could not write VERIFICATION_REPORT.tsv: {e})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
