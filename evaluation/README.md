# Evaluation

Every per-query record behind the reported results: 44,765 scored generations
and 16,260 LLM-judge decisions across 98 files.

`RESULTS.md` holds every figure. It is generated from the raw records, not
written by hand.

## What you can check, in ascending effort

**Read the results** — `RESULTS.md`. 661 table rows, all computed at build time.

**Regenerate them** (about a minute, no accelerator, no network):

    python3 -m pip install -r requirements-eval.txt
    cp RESULTS.md RESULTS.md.shipped
    python3 make_results.py
    diff RESULTS.md RESULTS.md.shipped

Every number in the document is recomputed from `results/*.jsonl`. A build-time
lint fails if any figure in the text cannot be traced to a computed value.

**Re-run individual generations** (needs the frozen system installed):

    python3 spotcheck.py --file 1a_A.jsonl -n 5
    python3 spotcheck.py --file 1a_B.jsonl -n 5
    python3 spotcheck.py --file 1a_B.jsonl --qid popqa-2

This picks stored records, re-executes them through the same code path that
produced them, and compares. Retrieval is checked separately from generation:
retrieved titles, sources and reranker logits must match exactly, and the
answer is checked against the stored one and against the gold answers.

## Layout

| path | contents |
|---|---|
| `RESULTS.md` | every figure, generated |
| `results/` | 98 files, one record per question per system |
| `sets/` | the frozen question sets, seeded and hashed |
| `manifests/` | per-run configuration, dataset hash, seed, timestamp |
| `make_results.py` | generates `RESULTS.md` |
| `make_final.py` | the statistics: loaders, paired bootstrap, McNemar, AUROC |
| `spotcheck.py` | re-runs stored generations and compares |
| `run_*.py` | the harness that produced each system |
| `run_ablations.sh` | the ablation grid |
| `common.py`, `freeze_sets.py` | shared library and set construction |

## Notes on reproduction

`spotcheck.py` covers the systems that ran locally. The Gemini comparisons were
produced through an external API and cannot be re-executed; their per-query
records are stored in full, and the document reports the measured run-to-run
variation of that API.

Retrieval is deterministic and reproduces exactly. Decode is deterministic
within a process, but the stored answers were produced mid-run, after hundreds
of prior questions in the same process, so a cold-start re-run of a single
question occasionally differs in wording while giving the same answer. Reported
accuracy is computed from the correctness verdict, not from the answer string.

The harnesses read the corpus and models at the path the frozen system expects,
which `install.py` in the parent directory sets up.

Internal identifiers (`A`, `B`, `N4`, `N5`, `N5p`, `N5e`, `G`, `C1`, `C2`,
`P2COL`) appear in `results/` and `manifests/` as the names each run was
executed under, and are preserved there unchanged so every figure traces back
to its records. `RESULTS.md` maps them to descriptive names.
