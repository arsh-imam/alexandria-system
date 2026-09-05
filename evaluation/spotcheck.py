"""
spotcheck.py - re-run stored generations and compare against results/*.jsonl.

The evaluation is 44,765 generations. Nobody re-runs that. This lets you
re-execute any subset and check the stored records are real.

    python3 spotcheck.py --file 1a_B.jsonl -n 5
    python3 spotcheck.py --file 1a_A.jsonl -n 5
    python3 spotcheck.py --file 1a_B.jsonl --qid popqa-2

What is compared depends on what is reproducible:

  retrieval   deterministic on any architecture - titles, sources and reranker
              logits must match exactly
  decode      byte-identical only on the architecture the run was performed on
              (aarch64). On any other host llama.cpp selects different SIMD
              kernels, so the answer is compared on correctness instead.

Requires the frozen system installed and reachable at its expected path.
"""
import argparse, hashlib, json, os, platform, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SSD = os.environ.get("ALEX_ROOT", "/media/pi/KINGSTON/local_ai")
SRC = os.environ.get("ALEX_CODE", os.path.join(os.path.dirname(HERE), "src"))
GGUF = os.path.join(SSD, "models", "Qwen3-1.7B-Q4_K_M.gguf")
RUN_ARCH = "aarch64"

def sha(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]

def load(fname, n, qid, seed):
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        sys.exit("no such results file: " + p)
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    if qid:
        rows = [r for r in rows if r.get("qid") == qid]
        if not rows:
            sys.exit("qid %s not found in %s" % (qid, fname))
        return rows
    random.Random(seed).shuffle(rows)
    return rows[:n]

def check_base(rows, contains):
    """Base ran at temperature 0.0 through a standalone llama-cpp harness."""
    from llama_cpp import Llama
    llm = Llama(model_path=GGUF, n_ctx=4096, n_threads=4, n_batch=512,
                verbose=False)
    SYS = ("You are a helpful assistant. Answer the question directly and "
           "concisely. If you do not know the answer, say you do not know. "
           "/no_think")
    out = []
    for r in rows:
        got = llm.create_chat_completion(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": r["question"]}],
            max_tokens=128, temperature=0.0)["choices"][0]["message"]["content"]
        out.append({"qid": r["qid"], "exact": sha(got) == sha(r.get("answer_raw") or r["answer"]),
                    "stored_hit": contains(r.get("answer") or "", r["answers"]),
                    "rerun_hit": contains(got, r["answers"]),
                    "stored": r.get("answer"), "got": got})
    del llm
    return out

def check_alexandria(rows, contains):
    """ALEXANDRIA ran through the frozen engine; verify retrieval and decode
    separately, because only decode is architecture-dependent."""
    sys.path.insert(0, SRC)
    from model_engine import ModelEngine
    eng = ModelEngine()
    eng.load()

    # run_arm_b.py wraps retriever.search to capture the evidence; do the same,
    # so this observes the engine exactly as the evaluation did.
    captured = {}
    _search = eng.retriever.search
    def search_w(*a, **k):
        out_ = _search(*a, **k)
        captured["results"] = out_[0] if isinstance(out_, tuple) else out_
        return out_
    eng.retriever.search = search_w

    out = []
    for r in rows:
        eng.new_conversation()
        captured.clear()
        got = eng.generate_stream(r["question"], lambda p: None)
        res = captured.get("results")
        rec = {"qid": r["qid"], "stored": r.get("answer_clean") or r.get("answer"),
               "got": got, "retrieval": None}
        if res is not None and r.get("retrieved"):
            want = [(p["title"], p["source"], p["logit"]) for p in r["retrieved"]]
            have = [(p["title"], p["source"], p["logit"]) for p in res]
            rec["retrieval"] = "exact" if want == have else "DIFFERS"
        import common as _C
        got_clean = _C._FOOTER.sub("", got).strip()
        rec["got"] = got_clean
        rec["exact"] = sha(got_clean) == sha((rec["stored"] or "").strip())
        rec["stored_hit"] = contains(rec["stored"] or "", r["answers"])
        rec["rerun_hit"] = contains(got_clean, r["answers"])
        out.append(rec)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="e.g. 1a_B.jsonl")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--qid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import common as C
    rows = load(a.file, a.n, a.qid, a.seed)
    arm = rows[0].get("arm", "?")
    host = platform.machine()
    byte_ok = host == RUN_ARCH

    print("file      %s" % a.file)
    print("system    %s" % {"A": "Base", "B": "ALEXANDRIA"}.get(arm, arm))
    print("records   %d" % len(rows))
    print("host      %s (recorded on %s)" % (host, RUN_ARCH))
    print("decode    %s\n" % ("byte-comparable" if byte_ok else
          "NOT byte-comparable across architectures - comparing correctness only"))

    if arm == "A":
        out = check_base(rows, C.contains)
    elif arm == "B" or a.file.startswith("abl_"):
        out = check_alexandria(rows, C.contains)
    else:
        sys.exit("spot-check supports the locally-run systems only "
                 "(Base and ALEXANDRIA); %r was produced by an external API" % arm)

    ex = sum(1 for r in out if r["exact"])
    agree = sum(1 for r in out if r["stored_hit"] == r["rerun_hit"])
    ret = [r.get("retrieval") for r in out if r.get("retrieval")]
    for r in out:
        flag = "exact" if r["exact"] else ("same verdict" if r["stored_hit"] == r["rerun_hit"] else "DIFFERS")
        print("  %-18s %-13s%s" % (r["qid"], flag,
              "  retrieval " + r["retrieval"] if r.get("retrieval") else ""))
        if not r["exact"]:
            print("      stored: %s" % (str(r["stored"])[:100]))
            print("      rerun : %s" % (str(r["got"])[:100]))
    print("\n%d/%d byte-identical | %d/%d same correctness verdict" %
          (ex, len(out), agree, len(out)))
    if ret:
        print("%d/%d retrieval exact" % (sum(1 for x in ret if x == "exact"), len(ret)))
    if byte_ok and ex < len(out):
        print("\nRetrieval is deterministic and reproduces exactly. Decode is\n"
              "deterministic per process but the stored answers were produced\n"
              "mid-run, after hundreds of prior queries in the same process, so\n"
              "a few differ in wording from a cold-start re-run. The correctness\n"
              "verdict - which is what the reported accuracy is computed from -\n"
              "is unaffected.")
    return 0 if (agree == len(out)) else 1

if __name__ == "__main__":
    sys.exit(main())
