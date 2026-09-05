"""Arm N4 - STANDARD 2026 RAG, properly implemented. THE baseline.

Pipeline (mainstream practice, e.g. LangChain/LlamaIndex + Cohere-style rerank):
  dense (Annoy/bge-small) + BM25 (FTS5, title-weighted)  -> pool 20 each
  Reciprocal Rank Fusion (k=60)
  cross-encoder rerank (ms-marco-MiniLM-L-6-v2 INT8 - SAME model arm B uses)
  dedup by title, take top-k=3
  neutral prompt permitting parametric fallback
NOT included (these are what arm B adds): federated ZIM deep-tier search,
entity-span title lookup, passage hygiene, authority prior, three-zone gate.
Same GGUF, same corpus, same hardware as arm B."""
import argparse, hashlib, json, os, re, sys, time
_ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENG not in sys.path: sys.path.insert(0, _ENG)
import common as C

SYS = ("You are a helpful assistant. Answer the question directly and "
       "concisely. Passages are provided that may be relevant; use them "
       "if they help, but if they do not contain the answer, answer from "
       "your own knowledge. If you do not know, say you do not know. /no_think")


def rrf(lists, k=60):
    sc = {}
    for lst in lists:
        for rank, pid in enumerate(lst):
            sc[pid] = sc.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--rerank-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import retriever as R
    from llama_cpp import Llama
    retr = R.Retriever()
    print("Loading index + reranker...", flush=True)
    retr.load()
    llm = Llama(model_path="/media/pi/KINGSTON/local_ai/models/Qwen3-1.7B-Q4_K_M.gguf",
                n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    print("Ready.\n", flush=True)

    cfg_hash = hashlib.md5(json.dumps(
        {"arm": "N4", "k": args.k, "pool": args.pool, "fusion": "RRF60",
         "dense": True, "bm25": True, "reranker": "ms-marco-MiniLM-L-6-v2-int8",
         "rerank_n": args.rerank_n, "dedup": "title", "gate": False,
         "deep_tier": False, "parametric_fallback": True},
        sort_keys=True).encode()).hexdigest()[:12]

    rows = [json.loads(l) for l in
            open(os.path.join(C.SETS, args.set), encoding="utf-8") if l.strip()]
    if args.limit: rows = rows[:args.limit]
    path = os.path.join(C.RES, args.out + ".jsonl")
    done = C.load_done(path)
    todo = [r for r in rows if r["qid"] not in done]
    print("%d in set | %d done | %d to run\n" % (len(rows), len(done), len(todo)), flush=True)

    C.write_manifest(args.out, {
        "arm": "N4", "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "k": args.k, "pool": args.pool,
        "rerank_n": args.rerank_n,
        "retrieval": "hybrid dense+BM25 -> RRF(60) -> cross-encoder rerank -> dedup -> top-3",
        "reranker": "ms-marco-MiniLM-L-6-v2 INT8 (identical to arm B)",
        "excluded_vs_B": ["federated ZIM deep tier", "entity-span title lookup",
                          "passage hygiene", "authority prior", "three-zone gate"],
        "system_prompt": SYS, "temperature": 0.0,
        "note": "STANDARD 2026 RAG baseline - the headline comparison arm"})

    w = C.Writer(path); t0all = time.time()
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        try:
            qv = retr.embedder.embed_one(r["question"])
            dense = list(retr.index.get_nns_by_vector(qv, args.pool))
            terms = R._content_terms(r["question"])
            if len(terms) > 3:
                terms = sorted(terms, key=retr._doc_freq)[:3]
            bm = []
            for joiner in (" AND ", " OR "):
                m = joiner.join('"%s"' % t for t in terms)
                try:
                    bm = [x["rowid"] for x in retr.db.execute(
                        "SELECT rowid FROM passages_fts WHERE passages_fts MATCH ? "
                        "ORDER BY bm25(passages_fts,10.0,1.0) LIMIT ?",
                        (m, args.pool)).fetchall()]
                except Exception:
                    bm = []
                if bm: break

            fused = rrf([dense, bm])[:args.rerank_n]
            cands = []
            if fused:
                qm = ",".join("?" * len(fused))
                by = {x["id"]: x for x in retr.db.execute(
                    "SELECT id,title,text,source FROM passages WHERE id IN (%s)" % qm,
                    fused).fetchall()}
                for pid in fused:
                    x = by.get(pid)
                    if x:
                        cands.append({"title": x["title"], "text": x["text"],
                                      "source": x["source"]})
            got = []
            if cands:
                logits = retr.reranker.score(
                    r["question"], ["%s — %s" % (c["title"], c["text"]) for c in cands])
                for c, lg in zip(cands, logits):
                    c["logit"] = round(float(lg), 3)
                cands.sort(key=lambda c: -c["logit"])
                seen = set()
                for c in cands:
                    if c["title"] in seen: continue
                    seen.add(c["title"]); got.append(c)
                    if len(got) >= args.k: break

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

        rec = {"qid": r["qid"], "arm": "N4", "config_hash": cfg_hash,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": r["question"],
               "answers": r.get("answers"), "answer": txt, "error": err,
               "n_retrieved": len(got), "retrieved": got,
               "top_logit": (got[0]["logit"] if got else None),
               "wall_s": round(time.time() - t0, 2)}
        for k in ("prop", "rel", "s_pop", "decile", "src", "subj"):
            if k in r: rec[k] = r[k]
        w.write(rec)
        print("  [%d/%d] %5.1fs eta %4.1fh  %s" %
              (i, len(todo), rec["wall_s"],
               (time.time()-t0all)/i*(len(todo)-i)/3600, r["question"][:42]), flush=True)
    w.close(); print("\nDone -> %s" % path)


if __name__ == "__main__":
    main()
