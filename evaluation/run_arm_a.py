"""Arm A - RAW Qwen3-1.7B. Minimal harness, NOT the engine.

Same GGUF, same n_ctx/n_threads/n_batch as the engine, but:
  - neutral prompt (plan sec 6), NOT ALEXANDRIA's hardened system prompt
  - greedy (temp 0)
  - " /no_think" appended to match the frozen engine's Qwen3 mode (V1)
  - <think> blocks stripped
  - no retrieval, no router, no compute/persona tier, no history
This is the honest 'raw base model' column. The old run_popqa.py raw arm
was the engine with retrieval off and is retired to context-anchor status.
"""
import argparse, hashlib, json, os, re, time
import common as C

GGUF = "/media/pi/KINGSTON/local_ai/models/Qwen3-1.7B-Q4_K_M.gguf"
SYS_FACTOID = ("You are a helpful assistant. Answer the question directly "
               "and concisely. If you do not know the answer, say you do "
               "not know.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    from llama_cpp import Llama
    sysmsg = SYS_FACTOID + " /no_think"
    cfg_hash = hashlib.md5(json.dumps(
        {"gguf": os.path.basename(GGUF), "sys": sysmsg, "temp": 0.0,
         "max_tokens": args.max_tokens, "n_ctx": 4096},
        sort_keys=True).encode()).hexdigest()[:12]

    print("Loading raw model (config %s)..." % cfg_hash, flush=True)
    llm = Llama(model_path=GGUF, n_ctx=4096, n_threads=4,
                n_batch=512, verbose=False)
    print("Loaded.\n", flush=True)

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
        "arm": "A", "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "gguf": GGUF, "system_prompt": sysmsg,
        "temperature": 0.0, "max_tokens": args.max_tokens,
        "n_ctx": 4096, "n_threads": 4, "n_batch": 512,
        "note": "raw base model; no retrieval, no engine scaffolding",
    })

    w = C.Writer(path)
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": r["question"]}],
                max_tokens=args.max_tokens, temperature=0.0, stream=False)
            raw = out["choices"][0]["message"]["content"]
            fin = out["choices"][0].get("finish_reason")
            ntok = out.get("usage", {}).get("completion_tokens")
            err = None
        except Exception as ex:
            raw, fin, ntok, err = "", None, None, "%s: %s" % (type(ex).__name__, ex)
        txt = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
        wall = round(time.time() - t0, 2)

        rec = {"qid": r["qid"], "arm": "A", "benchmark": args.set,
               "config_hash": cfg_hash,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "question": r["question"], "answers": r.get("answers"),
               "answer": txt, "answer_raw": raw, "had_think": bool(re.search(r"<think>\s*\S", raw or "")),  # non-EMPTY think block; Qwen3 emits an empty shell even under /no_think
               "finish_reason": fin, "output_tokens": ntok,
               "error": err, "wall_s": wall}
        for k in ("prop", "rel", "s_pop", "decile", "src", "subj"):
            if k in r:
                rec[k] = r[k]
        w.write(rec)

        el = time.time() - t_start
        eta = el / i * (len(todo) - i) / 3600.0
        print("  [%d/%d] %5.1fs eta %4.1fh  %s -> %s" %
              (i, len(todo), wall, eta, r["question"][:38], txt[:44].replace("\n", " ")),
              flush=True)
    w.close()
    print("\nDone -> %s" % path)


if __name__ == "__main__":
    main()
