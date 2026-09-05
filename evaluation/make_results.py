"""
make_results.py - generates RESULTS.md: every table and figure, no narrative.

Runs the computation in make_final.py unchanged, then emits only headings,
tables, and the caveats attached to them. No statistic is recomputed here;
this file changes presentation only.

Internal run identifiers (A, B, N4, N5, N5p, N5e, G, C1, C2, P2COL) are the
identifiers each system was run under, preserved verbatim in
results/*.jsonl and manifests/*.json. They are mapped to descriptive names
for display only.

    python3 make_results.py
"""
import os, re, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MF   = os.path.join(HERE, "make_final.py")
OUT  = os.path.join(HERE, "RESULTS.md")

# Longest patterns first: N5p must not be matched by the N5 rule.
DISPLAY = [
    (r"\bB coverage\b", "ALEXANDRIA coverage"),
    (r"\bArm A\b", "Base"),
    (r"\bPer-arm\b", "Per system"),
    (r"\bArm \(paired", "System (paired"),
    (r"oracle arm", "oracle condition"),
    (r"\bN_dense\b",            "Dense only"),
    (r"\bN_bm25\b",             "BM25 only"),
    (r"\(arm G\)", "(no evidence gate)"),
    (r"\barm[- ]([ABG])\b", lambda m: {"A":"Base","B":"ALEXANDRIA","G":"ALEXANDRIA (no gate)"}[m.group(1)]),
    (r"\bP2COL\b",              "Gemini 3.1 Pro"),
    (r"\bN5e\b",                "Oracle-Entity"),
    (r"\bN5p\b",                "Corpus-RAG + Entity"),
    (r"\bN5\b",                 "Corpus-RAG"),
    (r"\bN4\b",                 "Index-RAG"),
    (r"\bN3\b",                 "Hybrid-RAG"),
    (r"\bC2\b",                 "Gemini Flash + our context"),
    (r"\bC1\b",                 "Gemini 3.6 Flash"),
    (r"\barm A\b",              "Base"),
    (r"\barm B\b",              "ALEXANDRIA"),
    (r"\barm G\b",              "ALEXANDRIA (no gate)"),
    (r"\babl_nogate\b",         "no evidence gate"),
    (r"\babl_baseline\b",       "ablation baseline"),
    (r"\babl_deeponly(_\w+)?\b", "deep tier only"),
    (r"\babl_wiki_pure\b",      "Wikipedia only"),
    (r"\babl_wiki_deep\b",      "Wikipedia deep only"),
    (r"\babl_no([a-z0-9]+)\b", r"no \1"),
    (r"\babl_k(\d)\b",          r"final_k = \1"),
]

# Section titles phrased as claims, rewritten as neutral labels.
HEAD_RENAME = {
 "Retrieval performance is determined by whether the query identifies the entity":
   "Query construction and retrieval performance",
 "Does relation mix explain the arm-A anomaly?": "Relation mix, controlled",
 "Does bad evidence hurt? Matched-population test":
   "Matched-population test: effect of weak evidence",
 "Why: the gate withholds grounding when evidence is weak":
   "Grounding rate by evidence strength",
 "The evidence gate, causally tested (arm G)": "Evidence gate: causal test",
 "Frontier + our evidence (C2)": "Frontier model with our retrieved context",
 "A 1.7B model reading retrieved text vs a frontier model reading memory":
   "Local reader with evidence vs frontier closed-book",
 "Decomposed by whether B's passages contained the answer":
   "Decomposed by whether the retrieved passages contained the answer",
}

DISP_NAME = {"A": "Base", "B": "ALEXANDRIA", "G": "ALEXANDRIA (no gate)",
             "N4": "Index-RAG", "N5": "Corpus-RAG", "N5p": "Corpus-RAG + Entity",
             "N5e": "Oracle-Entity", "N3": "Hybrid-RAG",
             "C1": "Gemini 3.6 Flash", "C2": "Gemini Flash + our context",
             "P2COL": "Gemini 3.1 Pro",
             "N_dense": "Dense only", "N_bm25": "BM25 only"}

# make_final builds label cells as "CODE - NAME[CODE]"; collapse the whole cell.
_CELLNAME = re.compile(
    r"(\|\s*\*{0,2})(N_dense|N_bm25|N5p|N5e|N5|N4|N3|P2COL|C1|C2|A|B|G)"
    r"(\s*[\u2014-]\s*[^|]*?)(\*{0,2}\s*\|)")

CELL = {"A": "Base", "B": "ALEXANDRIA", "G": "ALEXANDRIA (no gate)"}

def relabel(s):
    if s.startswith("#"):
        s = re.sub(r"^(#+) F\d+\.\s*", r"\1 ", s)
        m = re.match(r"^(#+ )(?:(\d+)\. )?(.*)$", s)
        if m and m.group(3) in HEAD_RENAME:
            s = m.group(1) + (m.group(2) + ". " if m.group(2) else "") \
                + HEAD_RENAME[m.group(3)]
    s = re.sub("\\bB['\u2019]s\\b", "ALEXANDRIA's", s)
    if s.startswith("|"):
        s = _CELLNAME.sub(lambda m: m.group(1) + DISP_NAME[m.group(2)]
                          + m.group(4), s)
        s = re.sub(r"\|\s*Arm\s*\|", "| System |", s)
    for pat, rep in DISPLAY:
        s = re.sub(pat, rep, s)
    if s.startswith("|"):
        parts = s.split("|")
        for i, p in enumerate(parts):
            bare = p.strip().strip("*`_ ")
            if bare in CELL:
                parts[i] = p.replace(bare, CELL[bare])
        s = "|".join(parts)
    s = re.sub(r"(?<![\w`])([ABG]) (→|->) ([A-Za-z0-9_+ -]+)",
               lambda m: "%s %s %s" % (CELL.get(m.group(1), m.group(1)),
                                       m.group(2), m.group(3)), s)
    s = re.sub(r"([A-Za-z0-9_+ -]+) (→|->) ([ABG])(?![\w`])",
               lambda m: "%s %s %s" % (m.group(1), m.group(2),
                                       CELL.get(m.group(3), m.group(3))), s)
    return s

def is_table(l):   return l.startswith("|")
def is_head(l):    return l.startswith("#")
def is_quote(l):   return l.startswith(">")

def filter_lines(L):
    """Keep headings, tables, and the blockquote immediately following a table."""
    out, i, n = [], 0, len(L)
    while i < n:
        l = L[i]
        if is_head(l):
            if "FINAL EVALUATION RESULTS" not in l:
                out += ["", l, ""]
            i += 1; continue
        if is_table(l):
            while i < n and (is_table(L[i]) or (L[i] == "" and i+1 < n and is_table(L[i+1]))):
                out.append(L[i]); i += 1
            out.append("")
            j = i
            while j < n and L[j] == "": j += 1
            if j < n and is_quote(L[j]):
                while j < n and (is_quote(L[j]) or (L[j] == "" and j+1 < n and is_quote(L[j+1]))):
                    out.append(L[j]); j += 1
                out.append(""); i = j
            continue
        i += 1
    return out

def drop_empty_sections(body):
    heads = [i for i, l in enumerate(body) if l.startswith("#")]
    kill = set()
    for k, i in enumerate(heads):
        j = heads[k+1] if k+1 < len(heads) else len(body)
        if not any(x.startswith("|") for x in body[i+1:j]):
            kill.update(range(i, j))
    body = [l for i, l in enumerate(body) if i not in kill]
    cut = next((i for i, l in enumerate(body)
                if l.startswith("#") and "PART V" in l), None)
    if cut is not None:
        body = body[:cut]
    heads = [i for i, l in enumerate(body) if l.startswith("#")]
    drop = set()
    for k2, i in enumerate(heads):
        j = heads[k2+1] if k2+1 < len(heads) else len(body)
        if body[i].strip().lower() in ("## arms", "## suggested table mapping"):
            drop.update(range(i, j))
    return [l for i, l in enumerate(body) if i not in drop]


def renumber(body):
    out, k = [], 0
    for l in body:
        m = re.match(r"^## (\d+)\. (.*)$", l)
        if m:
            k += 1
            out.append("## %d. %s" % (k, m.group(2)))
        else:
            out.append(l)
    return out


def main():
    if not os.path.exists(MF):
        sys.exit("make_final.py not found next to this script")
    print("running the computation in make_final.py ...", flush=True)
    g = runpy.run_path(MF, run_name="_compute")
    L, F = g["L"], g["F"]
    print("  %d lines emitted, %d figures computed" % (len(L), len(F)))

    body = [relabel(x) for x in filter_lines(L)]
    body = renumber(drop_empty_sections(body))
    tables = sum(1 for x in body if x.startswith("|"))
    heads  = sum(1 for x in body if x.startswith("#"))

    head = [
        "# ALEXANDRIA - evaluation results",
        "",
        "Every figure below is computed directly from the per-query records in",
        "`results/*.jsonl` at build time. Nothing is transcribed.",
        "Regenerate with `python3 make_results.py`.",
        "",
        "**Scale:** %s scored generations and %s LLM-judge decisions."
        % (format(F.get("tot_gen", 0), ","), format(F.get("tot_judge", 0), ",")),
        "",
        "## Systems compared",
        "",
        "| name | description | internal id |",
        "|---|---|---|",
        "| Base | Qwen3-1.7B, no retrieval, greedy | `A` |",
        "| Index-RAG | RAG restricted to the pre-built index | `N4` |",
        "| Corpus-RAG | Full-corpus ZIM search, raw query | `N5` |",
        "| Corpus-RAG + Entity | adds entity parsing (deployable) | `N5p` |",
        "| Oracle-Entity | gold entity from benchmark metadata (upper bound, **not a baseline**) | `N5e` |",
        "| ALEXANDRIA | this work, frozen system as shipped | `B` |",
        "| ALEXANDRIA (no gate) | same passages, evidence gate bypassed | `G` |",
        "| Gemini 3.6 Flash | closed-book | `C1` |",
        "| Gemini Flash + our context | ALEXANDRIA's verbatim retrieved passages | `C2` |",
        "| Gemini 3.1 Pro | closed-book | `P2COL` |",
        "",
        "> Internal ids are the identifiers each system was run under. They appear",
        "> verbatim in `results/*.jsonl` and `manifests/*.json` and are preserved there",
        "> unchanged, so every table below can be traced to its raw records.",
        "",
        "---",
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(head + body).rstrip() + "\n")
    print("\nWROTE %s" % OUT)
    print("  %d lines | %d table rows | %d sections" % (len(head)+len(body), tables, heads))
    return body

def lint(body):
    doc = "\n".join(body)
    leftover = sorted(set(re.findall(r"(?<![\w`])(N[45][pe]?|P2COL|C[12]|abl_[a-z0-9_]+)(?![\w])", doc)))
    bare = sorted(set(re.findall(r"\|\s*\*{0,2}([ABG])\*{0,2}\s*\|", doc)))
    empty = [x for x in body if x.startswith("|") and set(x.replace("|","").strip()) <= {"-", " "} and "---" not in x]
    print("\nLINT")
    print("  unmapped labels : %s" % (leftover or "none"))
    print("  bare A/B/G cells: %s" % (bare or "none"))
    print("  malformed rows  : %d" % len(empty))
    if leftover or bare:
        print("  *** labels leaked through - fix DISPLAY before shipping ***")

if __name__ == "__main__":
    lint(main())
