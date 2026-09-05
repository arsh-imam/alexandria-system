"""Judge overlay (plan sec 4.2): semantic correctness + abstention class.
Scores every arm on the SAME items, symmetrically. Reports standard vs
semantic-adjusted accuracy; the gap is itself a finding."""
import argparse, json, os, random, time
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
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--n-disagree", type=int, default=300)
    ap.add_argument("--n-audit", type=int, default=300)
    args = ap.parse_args()

    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))

    KEY = {"B": "answer_clean"}
    arms = args.arms.split(",")
    D = {}
    for a in arms:
        p = os.path.join(C.RES, "%s_%s.jsonl" % (args.bench, a))
        if os.path.exists(p):
            D[a] = {r["qid"]: r for r in (json.loads(l) for l in
                    open(p, encoding="utf-8") if l.strip())}
    if not D:
        raise SystemExit("no arms found for " + args.bench)
    q = sorted(set.intersection(*[set(d) for d in D.values()]))
    print("arms: %s | paired n=%d" % (list(D), len(q)), flush=True)

    verdict = {a: {i: C.contains(D[a][i].get(KEY.get(a, "answer")) or "",
                                 D[a][i]["answers"]) for i in q} for a in D}
    disagree = [i for i in q if len(set(verdict[a][i] for a in D)) > 1]
    r = random.Random(C.SEED)
    pick = sorted(set(r.sample(disagree, min(args.n_disagree, len(disagree)))
                      + r.sample(q, min(args.n_audit, len(q)))))
    print("disagreement items: %d | judging %d items x %d arms = %d calls\n"
          % (len(disagree), len(pick), len(D), len(pick)*len(D)), flush=True)

    path = os.path.join(C.RES, "%s_judge.jsonl" % args.bench)
    done = C.load_done(path, key="key")
    w = C.Writer(path)
    todo = [(i, a) for i in pick for a in D if "%s|%s" % (i, a) not in done]
    t0 = time.time()
    for n, (i, a) in enumerate(todo, 1):
        rec = D[a][i]
        cand = (rec.get(KEY.get(a, "answer")) or "").strip()[:600]
        prompt = ("QUESTION: %s\nGOLD: %s\nCANDIDATE: %s"
                  % (rec["question"], "; ".join(rec["answers"][:6]), cand or "(empty)"))
        try:
            resp = cl.models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=40, system_instruction=RUBRIC))
            t = (resp.text or "")
            corr = "yes" in t.lower().split("correct:")[-1].split("\n")[0]
            comm = "yes" in t.lower().split("committed:")[-1].split("\n")[0]
            err = None
        except Exception as e:
            corr = comm = None; err = str(e)[:150]
        w.write({"key": "%s|%s" % (i, a), "qid": i, "arm": a,
                 "contains": verdict[a][i], "judge_correct": corr,
                 "judge_committed": comm, "error": err})
        if n % 50 == 0:
            print("  [%d/%d] %.1fs/call" % (n, len(todo), (time.time()-t0)/n), flush=True)
    w.close()

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print("\n" + "="*66)
    print("JUDGE OVERLAY — %s" % args.bench)
    print("="*66)
    print("  %-4s %10s %10s %12s %10s" % ("arm", "contains", "judge", "delta", "committed"))
    for a in D:
        s = [r for r in rows if r["arm"] == a and r["judge_correct"] is not None]
        if not s: continue
        c = sum(r["contains"] for r in s) / len(s)
        j = sum(r["judge_correct"] for r in s) / len(s)
        m = sum(r["judge_committed"] for r in s) / len(s)
        print("  %-4s %9.1f%% %9.1f%% %+11.1f %9.1f%%  (n=%d)"
              % (a, 100*c, 100*j, 100*(j-c), 100*m, len(s)))
    agree = [r for r in rows if r["judge_correct"] is not None]
    ok = sum(1 for r in agree if r["contains"] == r["judge_correct"])
    print("\n  judge/containment agreement: %.1f%% (n=%d)" % (100*ok/max(len(agree),1), len(agree)))
    print("  NOTE: 'delta' shows how much containment UNDER-counts each arm.")
