"""Arm N5 - single-query full-Wikipedia-ZIM RAG baseline.

CORRECTION (post-hoc): earlier versions of this docstring, and the run manifests
emitted by them, described N5 as an "equal-reach baseline" having "the SAME reach
as B". That is incorrect. N5 searches exactly one archive, the English Wikipedia
ZIM (see ZIM below). ALEXANDRIA searches 35 registered archives and additionally
draws candidates from the external passage index. N5 -> B is therefore an
aggregate difference in both retrieval architecture and corpus reach, not
architecture with reach held constant. Historical manifests retain the original
wording as evidence of what was actually run.

N5 uses archive-native full-text search over the Wikipedia ZIM with none of B's
retrieval architecture:
  - single raw query (no multi-query expansion, no rare-term or entity sub-queries)
  - no entity-span title lookup, no specialty routing, no authority prior, no hygiene
  - no three-zone gate (always injects top-k)
  - no fast-index generators
It DOES get: full-article chunking + cross-encoder rerank + title dedup, i.e. the
strongest reasonable naive implementation.
"""
import argparse, hashlib, html, json, os, re, sys, time
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.environ.get("ALEX_CODE", os.path.join(_REPO, "src"))
if not os.path.isdir(_SRC):          # historical layout: modules beside evaluation/
    _SRC = _REPO
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import common as C

ZIM = "/media/pi/KINGSTON/local_ai/wikipedia/wikipedia_en.zim"
SYS = ("You are a helpful assistant. Answer the question directly and "
       "concisely. Passages are provided that may be relevant; use them "
       "if they help, but if they do not contain the answer, answer from "
       "your own knowledge. If you do not know, say you do not know. /no_think")

_TAG = re.compile(r"<[^>]+>")
_SCR = re.compile(r"<(script|style).*?</\1>", re.S | re.I)

def strip_html(s):
    s = _SCR.sub(" ", s)
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def chunk(text, w=160, ov=40, cap=10):
    words = text.split()
    step = w - ov
    out = []
    for i in range(0, max(1, len(words)), step):
        piece = " ".join(words[i:i + w])
        if len(piece.split()) > 40:
            out.append(piece)
        if len(out) >= cap:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-per-title", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-gold-subj", action="store_true",
                    help="never use the benchmark's subj field; parse the question only")
    ap.add_argument("--query-mode", choices=["raw", "entity"], default="raw",
                    help="raw = user question verbatim; entity = subject-anchored query")
    args = ap.parse_args()

    import retriever as R
    from llama_cpp import Llama
    from libzim.reader import Archive
    from libzim.search import Searcher, Query

    print("Loading reranker + ZIM...", flush=True)
    retr = R.Retriever(); retr.load()
    arch = Archive(ZIM)
    llm = Llama(model_path="/media/pi/KINGSTON/local_ai/models/Qwen3-1.7B-Q4_K_M.gguf",
                n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    print("Ready.\n", flush=True)

    cfg_hash = hashlib.md5(json.dumps(
        {"arm": "N5", "reach": "full wikipedia_en.zim", "query": args.query_mode,
         "articles": args.articles, "chunks": args.chunks, "k": args.k,
         "reranker": "ms-marco-MiniLM-L-6-v2-int8", "gate": False,
         "multiquery": False, "entity_spans": False, "hygiene": False},
        sort_keys=True).encode()).hexdigest()[:12]

    rows = [json.loads(l) for l in
            open(os.path.join(C.SETS, args.set), encoding="utf-8") if l.strip()]
    if args.limit: rows = rows[:args.limit]
    path = os.path.join(C.RES, args.out + ".jsonl")
    done = C.load_done(path)
    todo = [r for r in rows if r["qid"] not in done]
    print("%d in set | %d done | %d to run\n" % (len(rows), len(done), len(todo)), flush=True)

    C.write_manifest(args.out, {
        "arm": "N5", "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "zim": ZIM,
        "retrieval": "single-query ZIM full-text search -> chunk -> cross-encoder -> dedup -> top-k",
        "excluded_vs_B": ["multi-query expansion", "entity-span title lookup",
                          "specialty routing", "authority prior", "passage hygiene",
                          "three-zone gate", "fast-index generators"],
        "query_mode": args.query_mode, "no_gold_subj": args.no_gold_subj,
        "note": "single-query full-Wikipedia-ZIM RAG baseline; searches the English Wikipedia ZIM only, not B's 35 registered archives",
        "system_prompt": SYS, "temperature": 0.0})

    w = C.Writer(path); t0all = time.time()
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        got = []
        try:
            cands = []
            if args.query_mode == "entity":
                # subject-anchored: use the benchmark's own subject string when
                # present, else strip interrogative scaffolding from the question.
                subj = None if args.no_gold_subj else r.get("subj")
                if not subj:
                    q_ = re.sub(r"^(who|what|when|where|which|in what)\s+", "", r["question"], flags=re.I)
                    q_ = re.sub(r"^(is|was|are|were|did|does|do)\s+", "", q_, flags=re.I)
                    q_ = re.sub(r"[?']s?\b|\?", " ", q_)
                    subj = " ".join(w for w in q_.split()
                                    if w.lower() not in ("the","a","an","of","for","in","by","did","play"))
                srch_q = subj
            else:
                srch_q = r["question"]
            try:
                res = Searcher(arch).search(Query().set_query(srch_q))
                paths = list(res.getResults(0, args.articles))
            except Exception:
                paths = []
            for p in paths:
                try:
                    e = arch.get_entry_by_path(p)
                    title = str(e.title)
                    body = strip_html(bytes(e.get_item().content).decode("utf-8", "ignore"))
                except Exception:
                    continue
                for ch in chunk(body, cap=args.chunks):
                    cands.append({"title": title, "text": ch, "source": "wikipedia (zim)"})
            if cands:
                logits = retr.reranker.score(
                    r["question"], ["%s — %s" % (c["title"], c["text"]) for c in cands])
                for c, lg in zip(cands, logits):
                    c["logit"] = round(float(lg), 3)
                cands.sort(key=lambda c: -c["logit"])
                seen = {}
                for c in cands:
                    n_ = seen.get(c["title"], 0)
                    if n_ >= args.max_per_title: continue
                    seen[c["title"]] = n_ + 1
                    got.append(c)
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
            txt, err = "", "%s: %s" % (type(ex).__name__, ex)

        rec = {"qid": r["qid"], "arm": "N5", "config_hash": cfg_hash,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": r["question"],
               "answers": r.get("answers"), "answer": txt, "error": err,
               "n_retrieved": len(got), "retrieved": got,
               "top_logit": (got[0]["logit"] if got else None),
               "n_candidates": len(cands) if 'cands' in dir() else 0,
               "search_query": srch_q if 'srch_q' in dir() else None,
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
