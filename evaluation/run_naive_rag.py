"""Arm N - NAIVE RAG baseline: what an obvious implementation looks like.
Same Pi, same Qwen3-1.7B, same corpus, same questions as arm B.
Differs ONLY in retrieval architecture:
  dense vectors only (no FTS5/BM25/deep/entity-spans)
  cosine ranking     (NO cross-encoder reranker)
  always inject top-3 (NO three-zone gate)
  neutral RAG prompt (NOT ALEXANDRIA's tuned prompts)
Engine files untouched: uses Retriever's loaded index read-only and calls
llama-cpp directly. Isolates 'our retrieval architecture' as the variable."""
import argparse, hashlib, json, os, re, sys, time
_ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)
import common as C

SYS = ("You are a helpful assistant. Answer the question directly and "
       "concisely using the passages provided. If you do not know the "
       "answer, say you do not know. /no_think")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", choices=["bm25", "dense"], default="bm25",
                    help="bm25 = realistic naive baseline; dense = weak-baseline control")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import retriever as R
    from llama_cpp import Llama

    retr = R.Retriever()
    print("Loading index (dense path only)...", flush=True)
    retr.load()

    llm = Llama(model_path="/media/pi/KINGSTON/local_ai/models/Qwen3-1.7B-Q4_K_M.gguf",
                n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    print("Ready.\n", flush=True)

    cfg_hash = hashlib.md5(json.dumps(
        {"arm": "N", "k": args.k, "mode": args.mode, "rank": "native", "reranker": False,
         "gate": False, "generators": ["vector"], "temp": 0.0}, sort_keys=True
    ).encode()).hexdigest()[:12]

    rows = [json.loads(l) for l in
            open(os.path.join(C.SETS, args.set), encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    path = os.path.join(C.RES, args.out + ".jsonl")
    done = C.load_done(path)
    todo = [r for r in rows if r["qid"] not in done]
    print("%d in set | %d done | %d to run\n" % (len(rows), len(done), len(todo)),
          flush=True)

    C.write_manifest(args.out, {
        "arm": "N", "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "k": args.k, "mode": args.mode,
        "retrieval": "dense Annoy only, cosine rank, no reranker, no gate",
        "system_prompt": SYS, "temperature": 0.0,
        "note": "naive-RAG baseline; isolates ALEXANDRIA's retrieval architecture",
    })

    w = C.Writer(path)
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        try:
            if args.mode == "dense":
                qv = retr.embedder.embed_one(r["question"])
                ids = retr.index.get_nns_by_vector(qv, args.k)
            else:
                terms = R._content_terms(r["question"])
                if len(terms) > 3:
                    terms = sorted(terms, key=retr._doc_freq)[:3]
                ids = []
                for joiner in (" AND ", " OR "):
                    m = joiner.join('"%s"' % t for t in terms)
                    try:
                        ids = [x["rowid"] for x in retr.db.execute(
                            "SELECT rowid FROM passages_fts WHERE passages_fts "
                            "MATCH ? ORDER BY bm25(passages_fts,10.0,1.0) LIMIT ?",
                            (m, args.k)).fetchall()]
                    except Exception:
                        ids = []
                    if ids:
                        break
            got = []
            if ids:
                qm = ",".join("?" * len(ids))
                by = {x["id"]: x for x in retr.db.execute(
                    "SELECT id,title,text,source FROM passages WHERE id IN (%s)" % qm,
                    ids).fetchall()}
                for pid in ids:
                    x = by.get(pid)
                    if x:
                        got.append({"title": x["title"], "text": x["text"],
                                    "source": x["source"]})
            ctx = "\n\n".join("[%s] %s" % (p["title"], " ".join(p["text"].split()[:120]))
                              for p in got)
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content":
                           "Passages:\n\n%s\n\nQuestion: %s" % (ctx, r["question"])}],
                max_tokens=128, temperature=0.0, stream=False)
            raw = out["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
            err = None
        except Exception as ex:
            txt, got, err = "", [], "%s: %s" % (type(ex).__name__, ex)
        rec = {"qid": r["qid"], "arm": "N", "config_hash": cfg_hash,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "question": r["question"], "answers": r.get("answers"),
               "answer": txt, "error": err,
               "n_retrieved": len(got), "retrieved": got,
               "wall_s": round(time.time() - t0, 2)}
        for k in ("prop", "rel", "s_pop", "decile", "src", "subj"):
            if k in r:
                rec[k] = r[k]
        w.write(rec)
        el = time.time() - t_start
        print("  [%d/%d] %5.1fs eta %4.1fh  %s" %
              (i, len(todo), rec["wall_s"], el/i*(len(todo)-i)/3600,
               r["question"][:42]), flush=True)
    w.close()
    print("\nDone -> %s" % path)


if __name__ == "__main__":
    main()
