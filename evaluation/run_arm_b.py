"""Arm B - the FROZEN ALEXANDRIA system, exactly as shipped.

Engine files are NEVER edited. All instrumentation is pure pass-through
observation wrappers installed at runtime:
  instrumentation.log_query   -> captures the engine's own per-query record
  model_engine.run_deliberate -> captures the FINAL evidence on multi-hop path
  <instance>.retriever.search -> captures evidence on the normal path
  <instance>.llm.create_chat_completion -> captures the EXACT prompt sent

Capturing the real prompt is what makes C2 honest: C2 replays the engine's
own context string verbatim (already 120-word-truncated by the engine),
never a reconstruction.
"""
import argparse, hashlib, json, os, subprocess, sys, time

# engine modules live one level up (~/local_ai_project); make them importable
_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

import common as C

SCRATCH = "/tmp/alex_eval_sessions"


def thermal():
    out = {"temp_c": None, "throttled": None}
    try:
        t = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                           text=True, timeout=5).stdout
        out["temp_c"] = float(t.split("=")[1].split("'")[0])
    except Exception:
        pass
    try:
        g = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                           text=True, timeout=5).stdout
        out["throttled"] = g.strip().split("=")[-1]
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="file in sets/")
    ap.add_argument("--out", required=True, help="output name in results/")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="shipped")
    ap.add_argument("--override", action="append", default=[],
                    help="retrieval/engine override, e.g. use_bm25=False")
    ap.add_argument("--fidelity", type=int, default=0,
                    help="run N queries with wrappers ON then OFF and compare")
    args = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)

    # ---- engine import + config (in-process only; no file is edited) ----
    import retriever as R
    R.RETRIEVAL_CONFIG["use_epistemic_fusion"] = True      # app behaviour
    import model_engine as ME
    import instrumentation as INSTR

    applied = {}
    for ov in args.override:
        k, v = ov.split("=", 1)
        val = {"True": True, "False": False}.get(v, v)
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                pass
        if k in R.RETRIEVAL_CONFIG:
            R.RETRIEVAL_CONFIG[k] = val; applied[k] = ("retrieval", val)
        elif k in ME.CONFIG:
            ME.CONFIG[k] = val; applied[k] = ("engine", val)
        else:
            raise SystemExit("unknown config key: " + k)

    ME.SESSIONS_DIR = SCRATCH                               # V3 redirect
    INSTR.LOG_FILE = os.path.join(C.RES, args.out + ".instr.jsonl")

    cfg_blob = json.dumps({"engine": ME.CONFIG, "retrieval": R.RETRIEVAL_CONFIG},
                          sort_keys=True, default=str)
    cfg_hash = hashlib.md5(cfg_blob.encode()).hexdigest()[:12]

    # ---- capture slots ----
    CAP = {}
    ON = {"v": True}

    def reset():
        CAP.clear()
        CAP.update(searches=[], final=None, calls=[], rec=None)

    _log = INSTR.log_query
    def log_w(rec):
        if ON["v"]:
            CAP["rec"] = dict(rec)
        return _log(rec)
    INSTR.log_query = log_w

    _rd = ME.run_deliberate
    def rd_w(*a, **k):
        out = _rd(*a, **k)
        if ON["v"]:
            CAP["final"] = out[0]
        return out
    ME.run_deliberate = rd_w

    eng = ME.ModelEngine()

    _search = eng.retriever.search
    def search_w(*a, **k):
        out = _search(*a, **k)
        if ON["v"]:
            CAP["searches"].append(out[0])
        return out
    eng.retriever.search = search_w

    print("Loading engine (config %s, tag=%s)..." % (cfg_hash, args.tag), flush=True)
    eng.load()

    _ccc = eng.llm.create_chat_completion
    def ccc_w(*a, **k):
        if ON["v"]:
            CAP["calls"].append(k.get("messages"))
        return _ccc(*a, **k)
    eng.llm.create_chat_completion = ccc_w
    print("Engine ready.\n", flush=True)

    # ---- question set ----
    rows = [json.loads(l) for l in
            open(os.path.join(C.SETS, args.set), encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    def ask(q):
        reset()
        t0 = time.time()
        try:
            ans = eng.generate_stream(q, lambda p: None)
            err = None
        except Exception as ex:
            ans, err = "", "%s: %s" % (type(ex).__name__, ex)
        wall = round(time.time() - t0, 2)
        diag = dict(getattr(eng.retriever, "last_diag", {}) or {})
        return ans, err, wall, diag

    # ---- fidelity mode ----
    if args.fidelity:
        sub = rows[:args.fidelity]
        print("FIDELITY: %d queries, wrappers ON then OFF.\n"
              "Retrieval is deterministic -> zone/top_logit MUST match exactly.\n"
              % len(sub), flush=True)
        recs = {}
        for phase in ("ON", "OFF"):
            ON["v"] = (phase == "ON")
            for i, r in enumerate(sub, 1):
                eng.new_conversation()
                ans, err, wall, diag = ask(r["question"])
                recs.setdefault(r["qid"], {})[phase] = {
                    "zone": (CAP.get("rec") or {}).get("zone") if phase == "ON" else None,
                    "top_logit": diag.get("top_logit"),
                    "pool": diag.get("pool_size"),
                    "winners": [w[0] for w in (diag.get("winners") or [])],
                    "wall": wall, "err": err,
                }
                print("  [%s %d/%d] %5.1fs logit=%s %s" %
                      (phase, i, len(sub), wall, diag.get("top_logit"),
                       r["question"][:44]), flush=True)
        bad = 0
        print("\n--- comparison ---")
        for qid, d in recs.items():
            a, b = d["ON"], d["OFF"]
            same = (a["top_logit"] == b["top_logit"]
                    and a["pool"] == b["pool"]
                    and a["winners"] == b["winners"])
            bad += (not same)
            if not same:
                print("  MISMATCH %s\n    ON =%s\n    OFF=%s" % (qid, a, b))
        print("\nmismatches: %d / %d" % (bad, len(recs)))
        print("VERDICT:", "WRAPPERS ARE TRANSPARENT - proceed."
              if bad == 0 else "**WRAPPERS ALTER BEHAVIOUR - STOP**")
        return

    # ---- normal run ----
    path = os.path.join(C.RES, args.out + ".jsonl")
    done = C.load_done(path)
    todo = [r for r in rows if r["qid"] not in done]
    print("%d in set | %d done | %d to run\n" % (len(rows), len(done), len(todo)),
          flush=True)
    C.write_manifest(args.out, {
        "arm": "B", "tag": args.tag, "set": args.set,
        "set_md5": C.md5(os.path.join(C.SETS, args.set)),
        "config_hash": cfg_hash, "overrides": applied,
        "engine_config": ME.CONFIG, "retrieval_config": R.RETRIEVAL_CONFIG,
        "n_total": len(rows),
    })

    w = C.Writer(path)
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        eng.new_conversation()
        ans, err, wall, diag = ask(r["question"])
        rec = CAP.get("rec") or {}

        ev = CAP.get("final")
        if ev is None:
            ev = CAP["searches"][0] if CAP["searches"] else []
        retrieved = [{"title": p.get("title"), "source": p.get("source"),
                      "logit": p.get("logit"), "score": p.get("score"),
                      "tier": p.get("tier"), "origin": p.get("origin"),
                      "text": p.get("text")} for p in ev]

        msgs = CAP["calls"][-1] if CAP["calls"] else None
        ctx = None
        if msgs:
            sysmsgs = [m["content"] for m in msgs[1:] if m.get("role") == "system"]
            ctx = sysmsgs[-1] if sysmsgs else None

        th = thermal()
        out = {
            "qid": r["qid"], "arm": "B", "tag": args.tag,
            "benchmark": args.set, "config_hash": cfg_hash,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": r["question"], "answers": r.get("answers"),
            "answer": ans, "answer_clean": C.strip_footer(ans), "error": err,
            "zone": rec.get("zone"), "path": rec.get("path"),
            "query_type": rec.get("query_type"),
            "deliberate": bool(rec.get("deliberate")),
            "sub_queries": rec.get("sub_queries"),
            "top_logit": diag.get("top_logit"),
            "separation": diag.get("separation"),
            "pool_size": diag.get("pool_size"),
            "routed": diag.get("routed"),
            "epistemic_need": diag.get("epistemic_need"),
            "n_retrieved": len(retrieved), "retrieved": retrieved,
            "context_block": ctx, "n_llm_calls": len(CAP["calls"]),
            "input_tokens": rec.get("input_tokens"),
            "output_tokens": rec.get("output_tokens"),
            "search_time": rec.get("search_time"),
            "decode_time": rec.get("decode_time"),
            "stage_timings": diag.get("timings"),
            "stage_timings_partial": bool(rec.get("deliberate")),
            "wall_s": wall, "temp_c": th["temp_c"], "throttled": th["throttled"],
        }
        for k in ("prop", "rel", "s_pop", "decile", "src", "subj"):
            if k in r:
                out[k] = r[k]
        w.write(out)

        el = time.time() - t_start
        eta = el / i * (len(todo) - i) / 3600.0
        print("  [%d/%d] %5.1fs %-10s logit=%-6s eta %4.1fh  %s" %
              (i, len(todo), wall, out["zone"], str(out["top_logit"]),
               eta, r["question"][:40]), flush=True)
    w.close()
    print("\nDone -> %s" % path)


if __name__ == "__main__":
    main()
