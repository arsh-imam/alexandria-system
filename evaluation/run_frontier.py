"""Frontier arms via Vertex/Gemini. C1 = closed-book; C2 = C1 + B's exact context.
Config locked from the T0 probe: default thinking ON, max_output_tokens=1500,
temp 0. Resumable; archives usage + finish_reason per call."""
import argparse, hashlib, json, os, time
import common as C

SYS_FACTOID = ("You are a helpful assistant. Answer the question directly "
               "and concisely. If you do not know the answer, say you do "
               "not know.")
C2_PREFIX = ("The following passages may help. Use them if relevant, and "
             "prefer them over memory for facts.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--arm", choices=["C1", "C2"], default="C1")
    ap.add_argument("--bfile", help="arm-B results file (required for C2)")
    ap.add_argument("--max-out", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True,
                          project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))

    rows = [json.loads(l) for l in
            open(os.path.join(C.SETS, args.set), encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    ctxmap = {}
    if args.arm == "C2":
        for l in open(os.path.join(C.RES, args.bfile + ".jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                ctxmap[r["qid"]] = r.get("context_block")

    cfg_hash = hashlib.md5(json.dumps(
        {"model": args.model, "arm": args.arm, "temp": 0,
         "max_out": args.max_out, "thinking": "default"},
        sort_keys=True).encode()).hexdigest()[:12]

    path = os.path.join(C.RES, args.out + ".jsonl")
    done = C.load_done(path)
    todo = [r for r in rows if r["qid"] not in done]
    print("%s | %s | %d in set | %d done | %d to run\n"
          % (args.arm, args.model, len(rows), len(done), len(todo)), flush=True)

    C.write_manifest(args.out, {
        "arm": args.arm, "model": args.model, "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "temperature": 0,
        "max_output_tokens": args.max_out, "thinking": "provider default (ON)",
        "access_date": time.strftime("%Y-%m-%d"),
        "system_prompt": SYS_FACTOID,
        "c2_prefix": C2_PREFIX if args.arm == "C2" else None,
        "b_source": args.bfile,
    })

    w = C.Writer(path)
    tt = to = th = 0
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        prompt = r["question"]
        has_ctx = False
        if args.arm == "C2":
            cb = ctxmap.get(r["qid"])
            if cb and "Reference notes:" in cb:
                body = cb.split("Reference notes:", 1)[1]
                body = body.rsplit("Answer the question using", 1)[0]
                body = body.rsplit("The reference notes below", 1)[0].strip()
                if body:
                    prompt = C2_PREFIX + body + "\n\nQuestion: " + r["question"]
                    has_ctx = True
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=args.max_out,
                    system_instruction=SYS_FACTOID))
            txt = (resp.text or "").strip()
            if not txt:                       # recover text from parts if .text is empty
                try:
                    for c_ in (resp.candidates or []):
                        for pt in (getattr(c_.content, "parts", None) or []):
                            if getattr(pt, "text", None):
                                txt += pt.text
                    txt = txt.strip()
                except Exception:
                    pass
            u = resp.usage_metadata
            fin = str(getattr(resp.candidates[0], "finish_reason", "?"))
            pt = u.prompt_token_count or 0
            ot = u.candidates_token_count or 0
            tk = getattr(u, "thoughts_token_count", 0) or 0
            tt += (u.total_token_count or 0); to += ot; th += tk
            err = None
        except Exception as ex:
            txt, fin, pt, ot, tk, err = "", None, 0, 0, 0, \
                "%s: %s" % (type(ex).__name__, ex)
            time.sleep(2)
        rec = {"qid": r["qid"], "arm": args.arm, "model": args.model,
               "config_hash": cfg_hash, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "question": r["question"], "answers": r.get("answers"),
               "answer": txt, "finish_reason": fin, "had_context": has_ctx,
               "prompt_tokens": pt, "output_tokens": ot, "thought_tokens": tk,
               "error": err, "wall_s": round(time.time() - t0, 2)}
        for k in ("prop", "rel", "s_pop", "decile", "src", "subj"):
            if k in r:
                rec[k] = r[k]
        w.write(rec)
        if i % 10 == 0 or i == len(todo):
            print("  [%d/%d] %.1fs/q | avg tok %d (out %d, thoughts %d)"
                  % (i, len(todo), (time.time()-t_start)/i, tt/i, to/i, th/i),
                  flush=True)
    w.close()
    rows_all = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    empty = sum(1 for r in rows_all if not (r.get("answer") or "").strip())
    print("\nDone -> %s\ntotal tokens: %d" % (path, tt))
    print("empty answers: %d/%d (%.1f%%)" % (empty, len(rows_all), 100*empty/max(len(rows_all),1)))
    if len(rows_all) and empty / len(rows_all) > 0.20:
        print("\n*** RUN FAILED: >20%% empty answers. DO NOT USE. ***")
        print("*** delete %s and investigate before re-running. ***" % path)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
