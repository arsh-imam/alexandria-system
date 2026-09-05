"""ALEXANDRIA evaluation - shared library.
Dataset loaders, the ONE global matcher (plan sec 4.2), seeded sampling,
resumable JSONL io, manifests. Imports NOTHING from the engine.
"""
import ast, hashlib, json, os, random, re, unicodedata
from datetime import datetime

SSD  = "/media/pi/KINGSTON/local_ai"
ROOT = os.path.dirname(os.path.abspath(__file__))
SETS = os.path.join(ROOT, "sets")
RES  = os.path.join(ROOT, "results")
MAN  = os.path.join(ROOT, "manifests")
SEED = 20260702

# ---------- the ONE global matcher ----------
_TOK = re.compile(r"[^\W_]+", re.UNICODE)

def _nfd(s):
    return unicodedata.normalize("NFD", s or "")

def toks(s, strip_accents=True):
    s = _nfd(s)
    if strip_accents:
        s = "".join(c for c in s if not unicodedata.combining(c))
    return _TOK.findall(s.lower())

def contains(text, golds, strip_accents=True):
    """True iff any gold alias appears as a contiguous WHOLE-TOKEN subsequence."""
    t = toks(text, strip_accents)
    if not t:
        return False
    n = len(t)
    for g in golds:
        gt = toks(g, strip_accents)
        m = len(gt)
        if m == 0 or m > n:
            continue
        for i in range(n - m + 1):
            if t[i:i + m] == gt:
                return True
    return False

_FOOTER = re.compile(
    r"\n*\s*(Sources:|Sources \(|Note: answered from model"
    r"|Computed exactly by ALEXANDRIA).*\Z", re.S)

def strip_footer(t):
    """Arm B cleanup: drop the auto-appended Sources/Note footer before matching."""
    return _FOOTER.sub("", t or "").rstrip()

# ---------- datasets ----------
def load_popqa():
    rows = json.load(open(os.path.join(SSD, "popqa.json"), encoding="utf-8"))
    out = []
    for i, r in enumerate(rows):
        a = r["answers"]
        if isinstance(a, str):
            a = ast.literal_eval(a)
        out.append({"qid": "popqa-%d" % i, "idx": i, "question": r["question"],
                    "answers": [str(x) for x in a], "s_pop": float(r["s_pop"]),
                    "prop": r["prop"], "subj": r["subj"], "obj": r["obj"]})
    return out

def popqa_longtail(rows=None):
    return [r for r in (rows or load_popqa()) if r["s_pop"] < 100]

def load_eq_test():
    d = os.path.join(SSD, "entityquestions", "dataset", "test")
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".test.json"):
            continue
        rel = fn.split(".")[0]
        data = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        out[rel] = [{"qid": "eq-%s-%d" % (rel, i), "rel": rel, "qi": i,
                     "question": q["question"],
                     "answers": [str(x) for x in q["answers"]]}
                    for i, q in enumerate(data)]
    return out

def eq_old_qids(path):
    """V7: recover the June-2026 1,200-question list as (rel, qi) pairs."""
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            out.append((r["rel"], r["qi"]))
    return out

# ---------- io ----------
def md5(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_done(path, key="qid"):
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    done.add(json.loads(ln)[key])
                except Exception:
                    pass
    return done

class Writer:
    """Append-mode, fsync'd per record -> a crash costs at most one query."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def rng():
    return random.Random(SEED)

def write_manifest(name, extra):
    os.makedirs(MAN, exist_ok=True)
    m = dict(run=name, utc=datetime.utcnow().isoformat() + "Z", seed=SEED)
    m.update(extra)
    p = os.path.join(MAN, "%s.json" % name)
    json.dump(m, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return p
