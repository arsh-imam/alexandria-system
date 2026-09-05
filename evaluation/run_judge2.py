"""Cross-check judge on a NON-GOOGLE model via any OpenAI-compatible endpoint.
Same rubric and same items as run_judge.py, so inter-judge agreement is exact.
Resumable. Prints running cost estimate."""
import argparse, collections, json, os, random, time
import common as C

RUBRIC = """You grade short factual answers. You are given a QUESTION, the GOLD answer(s), and a CANDIDATE answer.

Reply with EXACTLY two lines:
CORRECT: yes|no
COMMITTED: yes|no

CORRECT=yes if the candidate conveys the same fact as any gold answer, allowing synonyms, broader/narrower terms that a knowledgeable reader would accept, alternate spellings, and extra detail. CORRECT=no if it names a different entity, contradicts the gold, or gives no answer.
COMMITTED=yes if the candidate commits to a specific answer. COMMITTED=no if it declines, says it does not know, or merely lists several possibilities without choosing."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--arms", default="A,N4,B,C1,C2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--rate", type=float, default=0.0, help="seconds between calls")
    args = ap.parse_args()

    from openai import OpenAI
    cl = OpenAI(base_url=os.environ["JUDGE2_BASE_URL"],
                api_key=os.environ["JUDGE2_API_KEY"])

    KEY = {"B": "answer_clean"}
    D = {}
    for a in args.arms.split(","):
        p = os.path.join(C.RES, "%s_%s.jsonl" % (args.bench, a))
        if os.path.exists(p):
            D[a] = {r["qid"]: r for r in (json.loads(l) for l in
                    open(p, encoding="utf-8") if l.strip())}
    if not D:
        raise SystemExit("no arms found for " + args.bench)
    q = sorted(set.intersection(*[set(d) for d in D.values()]))

    # judge EXACTLY the items the Gemini judge already scored, for a paired comparison
    jp = os.path.join(C.RES, "%s_judge.jsonl" % args.bench)
    prior = set()
    if os.path.exists(jp):
        for l in open(jp, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("judge_correct") is not None:
                    prior.add(r["qid"])
    pool = sorted(prior & set(q)) or q
    pick = sorted(random.Random(C.SEED).sample(pool, min(args.items, len(pool))))
    print("arms=%s | paired pool=%d | judging %d items x %d arms = %d calls"
          % (list(D), len(pool), len(pick), len(D), len(pick) * len(D)), flush=True)

    path = os.path.join(C.RES, "%s_judge2.jsonl" % args.bench)
    done = C.load_done(path, key="key")
    todo = [(i, a) for i in pick for a in D if "%s|%s" % (i, a) not in done]
    print("%d calls remaining\n" % len(todo), flush=True)

    w = C.Writer(path)
    t0 = time.time(); tin = tout = 0; errs = 0
    for n, (i, a) in enumerate(todo, 1):
        rec = D[a][i]
        cand = (rec.get(KEY.get(a, "answer")) or "").strip()[:600]
        user = ("QUESTION: %s\nGOLD: %s\nCANDIDATE: %s"
                % (rec["question"], "; ".join(rec["answers"][:6]), cand or "(empty)"))
        corr = comm = None; err = None
        for attempt in range(4):
            try:
                r = cl.chat.completions.create(
                    model=args.model, temperature=0, max_tokens=30,
                    messages=[{"role": "system", "content": RUBRIC},
                              {"role": "user", "content": user}])
                t = (r.choices[0].message.content or "").lower()
                corr = "yes" in t.split("correct:")[-1].split("\n")[0]
                comm = "yes" in t.split("committed:")[-1].split("\n")[0]
                if r.usage:
                    tin += r.usage.prompt_tokens or 0
                    tout += r.usage.completion_tokens or 0
                err = None
                break
            except Exception as e:
                err = "%s: %s" % (type(e).__name__, str(e)[:120])
                time.sleep(2 ** attempt)
        if err: errs += 1
        w.write({"key": "%s|%s" % (i, a), "qid": i, "arm": a, "model": args.model,
                 "contains": C.contains(cand, rec["answers"]),
                 "judge_correct": corr, "judge_committed": comm, "error": err})
        if args.rate: time.sleep(args.rate)
        if n % 50 == 0 or n == len(todo):
            el = time.time() - t0
            print("  [%d/%d] %.2fs/call | tokens in %d out %d | errors %d | eta %.0f min"
                  % (n, len(todo), el/n, tin, tout, errs, (el/n)*(len(todo)-n)/60), flush=True)
    w.close()
    print("\nDONE. tokens: %d in, %d out | errors: %d" % (tin, tout, errs))
    if errs > len(todo) * 0.1:
        print("*** >10%% errors — inspect before using these results ***")

    # ---- inter-judge agreement ----
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = {(r["qid"], r["arm"]): r for r in rows if r.get("judge_correct") is not None}
    g = {}
    if os.path.exists(jp):
        for l in open(jp, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("judge_correct") is not None:
                    g[(r["qid"], r["arm"])] = r
    both = sorted(set(ok) & set(g))
    print("\n" + "=" * 76)
    print("INTER-JUDGE AGREEMENT — %s vs gemini-2.5-flash-lite (n=%d)" % (args.model, len(both)))
    print("=" * 76)
    if both:
        ag = sum(1 for k in both if ok[k]["judge_correct"] == g[k]["judge_correct"])
        p1 = sum(ok[k]["judge_correct"] for k in both) / len(both)
        p2 = sum(g[k]["judge_correct"] for k in both) / len(both)
        po = ag / len(both); pe = p1*p2 + (1-p1)*(1-p2)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print("  raw agreement  %.1f%%   Cohen's kappa %.3f" % (100*po, kappa))
        print("  positive rate: %s %.1f%% | gemini %.1f%%" % (args.model[:18], 100*p1, 100*p2))
        print("\n  %-5s %14s %14s %14s %10s" % ("arm", "containment", args.model[:12], "gemini", "agreement"))
        for a in sorted({k[1] for k in both}):
            s = [k for k in both if k[1] == a]
            print("  %-5s %13.1f%% %13.1f%% %13.1f%% %9.1f%%"
                  % (a, 100*sum(ok[k]["contains"] for k in s)/len(s),
                     100*sum(ok[k]["judge_correct"] for k in s)/len(s),
                     100*sum(g[k]["judge_correct"] for k in s)/len(s),
                     100*sum(1 for k in s if ok[k]["judge_correct"] == g[k]["judge_correct"])/len(s)))
        print("\n  kappa > 0.6 substantial | > 0.8 almost perfect")
        print("  If the two judges agree closely, the Gemini-judges-Gemini objection is answered.")


if __name__ == "__main__":
    main()
