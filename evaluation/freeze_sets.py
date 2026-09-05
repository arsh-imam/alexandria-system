"""Freeze every evaluation question set deterministically (seed 20260702).
Writes alex_eval/sets/*.jsonl + a manifest with source hashes.
Run once; re-running reproduces byte-identical files."""
import json, math, os, collections
import common as C

os.makedirs(C.SETS, exist_ok=True)

def dump(name, rows):
    p = os.path.join(C.SETS, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  %-26s n=%d" % (name, len(rows)))
    return p

def strat(rows, keyfn, total, r):
    """Proportional stratified sample, deterministic, largest-remainder."""
    groups = collections.defaultdict(list)
    for x in rows:
        groups[keyfn(x)].append(x)
    keys = sorted(groups)
    n = len(rows)
    exact = {k: len(groups[k]) * total / n for k in keys}
    take = {k: int(exact[k]) for k in keys}
    rem = total - sum(take.values())
    for k in sorted(keys, key=lambda k: -(exact[k] - int(exact[k])))[:rem]:
        take[k] += 1
    out = []
    for k in keys:
        g = sorted(groups[k], key=lambda x: x["qid"])
        t = min(take[k], len(g))
        out += r.sample(g, t) if t < len(g) else g
    return sorted(out, key=lambda x: x["qid"])

print("=== SOURCES ===")
pq_path = os.path.join(C.SSD, "popqa.json")
print("  popqa.json md5", C.md5(pq_path))

allpq = C.load_popqa()
lt = C.popqa_longtail(allpq)
print("  popqa rows %d | long-tail(<100) %d" % (len(allpq), len(lt)))

print("\n=== 1a HEADLINE ===")
dump("popqa_lt_1399.jsonl", sorted(lt, key=lambda x: x["idx"]))

print("\n=== PILOT (150 from 1a) ===")
pilot = sorted(C.rng().sample(sorted(lt, key=lambda x: x["qid"]), 150),
               key=lambda x: x["qid"])
dump("pilot_150.jsonl", pilot)

print("\n=== 1c ENTITYQUESTIONS (V7 exact reuse) ===")
eq = C.load_eq_test()
old = C.eq_old_qids(os.path.join(os.path.dirname(C.ROOT),
                                 "_results_eq_20260617", "eq_fed.jsonl"))
eqrows = [eq[rel][qi] for rel, qi in old]
assert len(eqrows) == 1200, len(eqrows)
dump("eq_1200.jsonl", eqrows)
print("  relations covered:", len(set(r["rel"] for r in eqrows)))

print("\n=== 1b POPULARITY DECILES (100 x 10) ===")
for r in allpq:
    r["logpop"] = math.log10(max(r["s_pop"], 1.0))
ordered = sorted(allpq, key=lambda x: (x["logpop"], x["qid"]))
rr = C.rng()
dec_rows, ltq = [], set(x["qid"] for x in lt)
size = len(ordered) / 10.0
reused = 0
for d in range(10):
    chunk = ordered[int(d * size):int((d + 1) * size)]
    pick = sorted(rr.sample(chunk, min(100, len(chunk))), key=lambda x: x["qid"])
    for x in pick:
        x2 = dict(x); x2["decile"] = d + 1
        reused += x2["qid"] in ltq
        dec_rows.append(x2)
    print("    decile %2d  s_pop %8.0f - %-8.0f  n=%d" %
          (d + 1, chunk[0]["s_pop"], chunk[-1]["s_pop"], len(pick)))
dump("popqa_deciles_1000.jsonl", dec_rows)
print("  overlap with 1a (reusable, no new Pi work): %d" % reused)

print("\n=== CORE-400 (ablation subset) ===")
r1 = C.rng()
a200 = strat(sorted(lt, key=lambda x: x["qid"]), lambda x: x["prop"], 200, r1)
b200 = strat(sorted(eqrows, key=lambda x: x["qid"]), lambda x: x["rel"], 200, r1)
for x in a200: x["src"] = "popqa_lt"
for x in b200: x["src"] = "eq"
dump("core400.jsonl", a200 + b200)
print("  1a props:", len(set(x["prop"] for x in a200)),
      "| eq rels:", len(set(x["rel"] for x in b200)))

man = C.write_manifest("sets_frozen", {
    "popqa_md5": C.md5(pq_path),
    "sets": {f: C.md5(os.path.join(C.SETS, f))
             for f in sorted(os.listdir(C.SETS)) if f.endswith(".jsonl")},
    "counts": {f: sum(1 for _ in open(os.path.join(C.SETS, f), encoding="utf-8"))
               for f in sorted(os.listdir(C.SETS)) if f.endswith(".jsonl")},
})
print("\nmanifest:", man)
print(open(man).read())
