# ALEXANDRIA - evaluation results

Every figure below is computed directly from the per-query records in
`results/*.jsonl` at build time. Nothing is transcribed.
Regenerate with `python3 make_results.py`.

**Scale:** 44,765 scored generations and 16,260 LLM-judge decisions.

## Systems compared

| name | description | internal id |
|---|---|---|
| Base | Qwen3-1.7B, no retrieval, greedy | `A` |
| Index-RAG | RAG restricted to the pre-built index | `N4` |
| Corpus-RAG | Full-corpus ZIM search, raw query | `N5` |
| Corpus-RAG + Entity | adds entity parsing (deployable) | `N5p` |
| Oracle-Entity | gold entity from benchmark metadata (upper bound, **not a baseline**) | `N5e` |
| ALEXANDRIA | this work, frozen system as shipped | `B` |
| ALEXANDRIA (no gate) | same passages, evidence gate bypassed | `G` |
| Gemini 3.6 Flash | closed-book | `C1` |
| Gemini Flash + our context | ALEXANDRIA's verbatim retrieved passages | `C2` |
| Gemini 3.1 Pro | closed-book | `P2COL` |

> Internal ids are the identifiers each system was run under. They appear
> verbatim in `results/*.jsonl` and `manifests/*.json` and are preserved there
> unchanged, so every table below can be traced to its raw records.

---

### Query construction and retrieval performance

| Configuration | Accuracy | Coverage | Acc when covered |
|---|---|---|---|
| Base | 24.8% | — | — |
| Index-RAG | 16.7% | 27.1% | 51.7% |
| Corpus-RAG | 24.4% | 22.9% | 65.1% |
| Corpus-RAG + Entity* | 37.4% | 41.7% | 79.9% |
| **ALEXANDRIA** | **45.6%** | **49.5%** | **81.8%** |
| *Oracle-Entity* | *59.3%* | *68.0%* | *83.6%* |

| Step | Isolates | Δ | p |
|---|---|---|---|
| Base → Index-RAG | index truncation | -8.1 pp | 2.34e-15 |
| Base → Corpus-RAG | corpus reach, naive query | -0.4 pp | 7.33e-01 (ns) |
| Corpus-RAG → Corpus-RAG + Entity | entity parsing | +13.0 pp | 6.15e-33 |
| **Corpus-RAG + Entity → ALEXANDRIA** | **ALEXANDRIA's retrieval architecture** | **+8.2 pp** | **1.04e-09** |
| *Corpus-RAG + Entity → Oracle-Entity* | *gold entity metadata (headroom)* | *+21.9 pp* | *6.32e-56* |
| Base → ALEXANDRIA | full system vs no retrieval | +20.8 pp | 2.56e-48 |

| Stratum | Reach alone (Base → Corpus-RAG) | p | Architecture (Corpus-RAG + Entity → ALEXANDRIA) | p |
|---|---|---|---|---|
| **Small answer space** (country, capital, capital of, sport, religion, occupation) — 47.5% | -5.7 pp | 1.42e-03 | +11.0 pp | 4.30e-07 |
| **Creative-work attribution** (author, director, composer, screenwriter, producer, genre) — 38.1% | +8.4 pp | 7.05e-11 | -3.0 pp | 6.81e-02 (ns) |
| **Other biographical** (place of birth, father, mother) — 14.4% | -6.4 pp | 1.46e-02 | +28.9 pp | 2.05e-11 |
| *aggregate (what the pooled table shows)* | *-0.4 pp* | *7.33e-01 (ns)* | *+8.2 pp* | *1.04e-09* |


## System under test

| Component | Size | Snapshot | Registered for retrieval |
|---|---|---|---|
| `wikipedia_en.zim` | 51.93 GB | 2026-03-19 | **yes** |
| `zim_extra/` — 34 specialty archives | 13.37 GB | 2026-02 to 2026-05 | **yes** |
| `wiktionary_en.zim` | 9.12 GB | 2026-05-11 | no |
| `ifixit_en.zim` | 3.57 GB | 2025-12-21 | no |
| `wikibooks_en.zim` | 3.51 GB | 2026-04-27 | no |
| `wikiquote_en.zim` | 0.32 GB | 2026-04-14 | no |
| `wikivoyage_en.zim` | 0.23 GB | 2026-03-17 | no |
| `wikinews_en.zim` | 0.07 GB | 2026-04-14 | no |
| **Searched total** | **65.30 GB** | — | **35 archives** |
| On disk total | 82.12 GB | — | 41 archives |


## Benchmarks

| ID | Benchmark | n | Construction |
|---|---|---|---|
| 1a | PopQA long-tail (`s_pop < 100`) | 1399 | Full subset; count matches published usage |
| 1b | PopQA popularity deciles | 1000 | 100 per log-pageview decile from all 14,267 |
| 1c | EntityQuestions | 1200 | Relation-stratified, 50 × 24 |
| 1d | NQ-Open | 500 | Uniform sample of the 3,610 validation split |
| core-400 | 1a + 1c stratified mix | 400 | 200 from each, used for ablations |


## External anchoring — published results on the identical subset

| System | Accuracy | Source |
|---|---|---|
| Llama2-7B | 14.7% | Self-RAG, Table 2 |
| Llama2-13B | 14.7% | Self-RAG, Table 2 |
| Alpaca-7B | 23.6% | Self-RAG, Table 2 |
| Alpaca-13B | 24.4% | Self-RAG, Table 2 |
| **Qwen3-1.7B (this work, Base)** | **24.8%** | **this work** |
| ChatGPT | 29.3% | Self-RAG, Table 2 |

| System | Accuracy | Source |
|---|---|---|
| Llama2-7B + RAG | 38.2% | Self-RAG, Table 2 |
| **ALEXANDRIA — 1.7B, CPU-only, fully offline (this work, ALEXANDRIA)** | **45.6%** | **this work** |
| Llama2-13B + RAG | 45.7% | Self-RAG, Table 2 |
| Alpaca-13B + RAG | 46.1% | Self-RAG, Table 2 |
| Alpaca-7B + RAG | 46.7% | Self-RAG, Table 2 |
| Llama2-FT-7B + RAG | 48.7% | Self-RAG, Table 2 |
| LLaMA2-7B + plain RAG | 50.5% | CRAG (Yan et al. 2024) |
| Ret-ChatGPT | 50.8% | Self-RAG, Table 2 |
| Ret-Llama2-chat-13B | 51.8% | Self-RAG, Table 2 |
| SelfRAG-LLaMA2-7B + RAG | 52.8% | CRAG (Yan et al. 2024) |
| Self-RAG 7B | 54.9% | Self-RAG, Table 2 |
| Self-RAG 13B | 55.8% | Self-RAG, Table 2 |
| Self-CRAG 7B | 61.8% | CRAG (Yan et al. 2024) |


## Pre-run validity checks

| Check | Result |
|---|---|
| Embedding alignment (`self_cos`, 4 probes; target ≥ 0.98) | **1.0000** |
| Annoy ↔ SQLite ID alignment (passage retrieves itself at rank 1) | **3/3** |
| Instrumentation fidelity (wrappers ON vs OFF, 20 queries) | **0/20 mismatches** |
| Matcher unit tests | **7/7** |
| Frontier output cap (150 measured to truncate; raised to 1500) | **0.4% truncated** |
| Thinking-mode match (engine `/no_think`; Base pinned identically) | 12 vs 128 tokens |


### 1a — PopQA long-tail (paired n = 1399)

| System | Accuracy | Hits | 95% CI |
|---|---|---|---|
| Base | 24.8% | 347 | [22.7–27.0] |
| Index-RAG | 16.7% | 233 | [14.7–18.7] |
| Corpus-RAG | 24.4% | 341 | [22.2–26.6] |
| ALEXANDRIA | 45.6% | 638 | [43.0–48.2] |
| Gemini 3.6 Flash | 58.8% | 822 | [56.2–61.3] |
| Gemini Flash + our context | 59.4% | 831 | [56.8–62.0] |
| Gemini 3.1 Pro | 60.1% | 841 | [57.5–62.7] |
| *Dense only* | *11.7%* | *163* | *[10.0–13.4]* |
| *BM25 only* | *12.5%* | *175* | *[10.8–14.2]* |

| Comparison | Δ | Discordant | p |
|---|---|---|---|
| Index-RAG → ALEXANDRIA | +28.9 pp | 35 / 440 | 2.88e-90 |
| Corpus-RAG → ALEXANDRIA | +21.2 pp | 57 / 354 | 1.84e-53 |
| Base → ALEXANDRIA | +20.8 pp | 70 / 361 | 2.56e-48 |
| ALEXANDRIA → Gemini 3.6 Flash | +13.2 pp | 110 / 294 | 1.84e-20 |
| ALEXANDRIA → Gemini 3.1 Pro | +14.5 pp | 100 / 303 | 7.61e-25 |
| Base → Index-RAG | -8.1 pp | 164 / 50 | 2.34e-15 |
| Gemini 3.6 Flash → Gemini Flash + our context | +0.6 pp | 130 / 139 | 6.26e-01 (ns) |


### 1b — PopQA popularity deciles (paired n = 1000)

| System | Accuracy | Hits | 95% CI |
|---|---|---|---|
| Base | 15.4% | 154 | [13.2–17.6] |
| Index-RAG | 22.2% | 222 | [19.7–24.8] |
| Corpus-RAG | 28.8% | 288 | [26.1–31.6] |
| ALEXANDRIA | 37.2% | 372 | [34.3–40.2] |
| Gemini 3.6 Flash | 73.9% | 739 | [71.1–76.6] |
| Gemini Flash + our context | 62.0% | 620 | [59.0–65.0] |
| Gemini 3.1 Pro | 74.5% | 745 | [71.8–77.1] |

| Comparison | Δ | Discordant | p |
|---|---|---|---|
| Index-RAG → ALEXANDRIA | +15.0 pp | 50 / 200 | 1.98e-22 |
| Corpus-RAG → ALEXANDRIA | +8.4 pp | 84 / 168 | 1.32e-07 |
| Base → ALEXANDRIA | +21.8 pp | 29 / 247 | 2.81e-44 |
| ALEXANDRIA → Gemini 3.6 Flash | +36.7 pp | 20 / 387 | 2.54e-89 |
| ALEXANDRIA → Gemini 3.1 Pro | +37.3 pp | 17 / 390 | 2.92e-93 |
| Base → Index-RAG | +6.8 pp | 46 / 114 | 7.59e-08 |
| Gemini 3.6 Flash → Gemini Flash + our context | -11.9 pp | 159 / 40 | 5.43e-18 |


### 1c — EntityQuestions (paired n = 1200)

| System | Accuracy | Hits | 95% CI |
|---|---|---|---|
| Base | 18.0% | 216 | [15.8–20.2] |
| Index-RAG | 22.4% | 269 | [20.1–24.8] |
| ALEXANDRIA | 52.5% | 630 | [49.7–55.3] |
| Gemini 3.6 Flash | 74.6% | 895 | [72.1–77.1] |
| Gemini Flash + our context | 71.7% | 860 | [69.1–74.2] |
| Gemini 3.1 Pro | 73.2% | 878 | [70.7–75.7] |

| Comparison | Δ | Discordant | p |
|---|---|---|---|
| Index-RAG → ALEXANDRIA | +30.1 pp | 40 / 401 | 4.63e-76 |
| Base → ALEXANDRIA | +34.5 pp | 25 / 439 | 6.84e-99 |
| ALEXANDRIA → Gemini 3.6 Flash | +22.1 pp | 63 / 328 | 2.54e-44 |
| ALEXANDRIA → Gemini 3.1 Pro | +20.7 pp | 69 / 317 | 4.36e-39 |
| Base → Index-RAG | +4.4 pp | 64 / 117 | 1.00e-04 |
| Gemini 3.6 Flash → Gemini Flash + our context | -2.9 pp | 120 / 85 | 1.74e-02 |


### 1d — NQ-Open (paired n = 500)

| System | Accuracy | Hits | 95% CI |
|---|---|---|---|
| Base | 18.4% | 92 | [15.0–21.8] |
| Index-RAG | 33.0% | 165 | [29.0–37.2] |
| Corpus-RAG | 36.4% | 182 | [32.2–40.6] |
| ALEXANDRIA | 39.0% | 195 | [34.8–43.4] |
| Gemini 3.6 Flash | 68.2% | 341 | [64.0–72.2] |
| Gemini Flash + our context | 65.6% | 328 | [61.4–69.8] |
| Gemini 3.1 Pro | 66.0% | 330 | [61.8–70.2] |
| *BM25 only* | *23.4%* | *117* | *[19.8–27.2]* |

| Comparison | Δ | Discordant | p |
|---|---|---|---|
| Index-RAG → ALEXANDRIA | +6.0 pp | 42 / 72 | 6.35e-03 |
| Corpus-RAG → ALEXANDRIA | +2.6 pp | 50 / 63 | 2.59e-01 (ns) |
| Base → ALEXANDRIA | +20.6 pp | 28 / 131 | 3.95e-17 |
| ALEXANDRIA → Gemini 3.6 Flash | +29.2 pp | 22 / 168 | 4.99e-29 |
| ALEXANDRIA → Gemini 3.1 Pro | +27.0 pp | 23 / 158 | 5.78e-26 |
| Base → Index-RAG | +14.6 pp | 22 / 95 | 5.31e-12 |
| Gemini 3.6 Flash → Gemini Flash + our context | -2.6 pp | 39 / 26 | 1.36e-01 (ns) |


### Multiple-comparison correction

| Rank | Test | p | Threshold | Verdict |
|---|---|---|---|---|
| 1 | 1c Base → ALEXANDRIA | 6.84e-99 | 0.0042 | SIG |
| 2 | 1a Index-RAG → ALEXANDRIA | 2.88e-90 | 0.0045 | SIG |
| 3 | 1b ALEXANDRIA → Gemini 3.6 Flash | 2.54e-89 | 0.0050 | SIG |
| 4 | 1c Index-RAG → ALEXANDRIA | 4.63e-76 | 0.0056 | SIG |
| 5 | 1a Base → ALEXANDRIA | 2.56e-48 | 0.0063 | SIG |
| 6 | 1c ALEXANDRIA → Gemini 3.6 Flash | 2.54e-44 | 0.0071 | SIG |
| 7 | 1b Base → ALEXANDRIA | 2.81e-44 | 0.0083 | SIG |
| 8 | 1d ALEXANDRIA → Gemini 3.6 Flash | 4.99e-29 | 0.0100 | SIG |
| 9 | 1b Index-RAG → ALEXANDRIA | 1.98e-22 | 0.0125 | SIG |
| 10 | 1a ALEXANDRIA → Gemini 3.6 Flash | 1.84e-20 | 0.0167 | SIG |
| 11 | 1d Base → ALEXANDRIA | 3.95e-17 | 0.0250 | SIG |
| 12 | 1d Index-RAG → ALEXANDRIA | 6.35e-03 | 0.0500 | SIG |


## 1. The query-construction ladder

| Configuration | Accuracy | 95% CI | Coverage | Acc when covered |
|---|---|---|---|---|
| Base | 24.8% | [22.7–27.0] | — | — |
| Index-RAG | 16.7% | [14.7–18.7] | 27.1% | 51.7% |
| Corpus-RAG | 24.4% | [22.2–26.6] | 22.9% | 65.1% |
| Corpus-RAG + Entity | 37.4% | [34.9–39.9] | 41.7% | 79.9% |
| **ALEXANDRIA** | 45.6% | [43.0–48.2] | 49.5% | 81.8% |
| *Oracle-Entity* | 59.3% | [56.7–61.8] | 68.0% | 83.6% |

| Step | Δ | Discordant | p |
|---|---|---|---|
| Base → Index-RAG  index truncation | -8.1 pp | 164 / 50 | 2.34e-15 |
| Base → Corpus-RAG  corpus reach, naive query | -0.4 pp | 110 / 104 | 7.33e-01 (ns) |
| Corpus-RAG → Corpus-RAG + Entity  entity parsing | +13.0 pp | 36 / 218 | 6.15e-33 |
| **Corpus-RAG + Entity → ALEXANDRIA  ALEXANDRIA architecture** | +8.2 pp | 120 / 235 | 1.04e-09 |
| *Corpus-RAG + Entity → Oracle-Entity  gold metadata (headroom)* | +21.9 pp | 56 / 362 | 6.32e-56 |
| Base → ALEXANDRIA  full system vs no retrieval | +20.8 pp | 70 / 361 | 2.56e-48 |

> **Oracle-Entity is an oracle condition and must never be reported as a baseline.** It receives the benchmark's
> `subj` field — gold entity metadata that no deployed system possesses. It is included solely to
> bound the headroom available from perfect entity identification. The deployable baseline is Corpus-RAG + Entity.


## 2. Relation-stratified ladder

| Relation | n | Base | Corpus-RAG | Corpus-RAG + Entity | ALEXANDRIA | Oracle-Entity | Base → Corpus-RAG | Corpus-RAG + Entity → ALEXANDRIA | ALEXANDRIA coverage |
|---|---|---|---|---|---|---|---|---|---|
| capital | 29 | 65.5% | 65.5% | 86.2% | 82.8% | 79.3% | +0.0 (ns) | -3.4 (ns) | 75.9% |
| country | 274 | 56.6% | 61.7% | 83.2% | 77.7% | 85.4% | +5.1 (ns) | -5.5 | 73.0% |
| sport | 196 | 52.6% | 37.2% | 41.8% | 66.3% | 86.2% | -15.4 | +24.5 | 70.4% |
| occupation | 121 | 23.1% | 9.1% | 31.4% | 59.5% | 76.9% | -14.0 | +28.1 | 62.0% |
| religion | 40 | 17.5% | 5.0% | 22.5% | 37.5% | 27.5% | -12.5 (ns) | +15.0 (ns) | 57.5% |
| place of birth | 173 | 14.5% | 4.6% | 12.7% | 46.2% | 75.7% | -9.9 | +33.5 | 58.4% |
| genre | 113 | 7.1% | 8.0% | 22.1% | 15.9% | 24.8% | +0.9 (ns) | -6.2 (ns) | 20.4% |
| father | 27 | 3.7% | 18.5% | 40.7% | 40.7% | 59.3% | +14.8 (ns) | +0.0 (ns) | 48.1% |
| composer | 48 | 2.1% | 10.4% | 20.8% | 14.6% | 22.9% | +8.3 (ns) | -6.2 (ns) | 16.7% |
| author | 198 | 0.0% | 16.7% | 28.8% | 22.2% | 35.4% | +16.7 | -6.6 | 26.3% |
| producer | 37 | 0.0% | 2.7% | 13.5% | 8.1% | 13.5% | +2.7 (ns) | -5.4 (ns) | 18.9% |
| director | 95 | 0.0% | 6.3% | 7.4% | 12.6% | 27.4% | +6.3 | +5.2 (ns) | 20.0% |
| screenwriter | 42 | 0.0% | 0.0% | 4.8% | 14.3% | 23.8% | +0.0 (ns) | +9.5 (ns) | 23.8% |


### Stratification rule

| Stratum | Relations | n | Share |
|---|---|---|---|
| **Small answer space** | country, capital, capital of, sport, religion, occupation | 665 | 47.5% |
| **Creative-work attribution** | author, director, composer, screenwriter, producer, genre | 533 | 38.1% |
| **Other biographical** | place of birth, father, mother | 201 | 14.4% |


### Sensitivity of both headline effects to the stratum boundary

| Boundary | n small | reach | p | arch | p | n hard | reach | p | arch | p |
|---|---|---|---|---|---|---|---|---|---|---|
| creative-work only in second stratum (14.4% unassigned) | 665 | -5.7 pp | 1.42e-03 | +11.0 pp | 4.30e-07 | 533 | +8.4 pp | 7.05e-11 | -3.0 pp | 6.81e-02 (ns) |
| + place of birth | 665 | -5.7 pp | 1.42e-03 | +11.0 pp | 4.30e-07 | 706 | +4.0 pp | 1.52e-03 | +6.0 pp | 5.34e-04 |
| + all biographical (COMPLETE, used throughout) | 665 | -5.7 pp | 1.42e-03 | +11.0 pp | 4.30e-07 | 734 | +4.3 pp | 3.78e-04 | +5.8 pp | 6.98e-04 |
| COMPLETE, but `country` moved out of small-answer-space | 391 | -13.3 pp | 1.35e-08 | +22.5 pp | 1.12e-12 | 1008 | +4.6 pp | 5.88e-05 | +2.7 pp | 6.37e-02 (ns) |


### Strata compared

| System | Accuracy | Coverage | Acc when covered |
|---|---|---|---|
| Base | 46.9% | — | — |
| Index-RAG | 32.5% | 50.4% | 54.0% |
| Corpus-RAG | 41.2% | 33.8% | 67.1% |
| Corpus-RAG + Entity | 57.7% | 60.5% | 82.8% |
| ALEXANDRIA | 68.7% | 69.2% | 86.5% |
| Oracle-Entity | 80.0% | 85.3% | 88.2% |

| Step | Δ | p |
|---|---|---|
| Base → Corpus-RAG  reach alone | -5.7 pp | 1.42e-03 |
| Corpus-RAG → Corpus-RAG + Entity  entity parsing | +16.5 pp | 1.14e-18 |
| Corpus-RAG + Entity → ALEXANDRIA  ALEXANDRIA architecture | +11.0 pp | 4.30e-07 |
| Base → ALEXANDRIA  full system | +21.8 pp | 7.19e-21 |

| System | Accuracy | Coverage | Acc when covered |
|---|---|---|---|
| Base | 1.7% | — | — |
| Index-RAG | 2.1% | 4.7% | 44.0% |
| Corpus-RAG | 10.1% | 12.9% | 66.7% |
| Corpus-RAG + Entity | 19.9% | 24.8% | 76.5% |
| ALEXANDRIA | 16.9% | 22.3% | 66.4% |
| Oracle-Entity | 28.1% | 41.8% | 66.8% |

| Step | Δ | p |
|---|---|---|
| Base → Corpus-RAG  reach alone | +8.4 pp | 7.05e-11 |
| Corpus-RAG → Corpus-RAG + Entity  entity parsing | +9.8 pp | 3.07e-12 |
| Corpus-RAG + Entity → ALEXANDRIA  ALEXANDRIA architecture | -3.0 pp | 6.81e-02 (ns) |
| Base → ALEXANDRIA  full system | +15.2 pp | 1.42e-21 |

| System | Accuracy | Coverage | Acc when covered |
|---|---|---|---|
| Base | 12.9% | — | — |
| Index-RAG | 3.0% | 9.5% | 21.1% |
| Corpus-RAG | 6.5% | 13.4% | 44.4% |
| Corpus-RAG + Entity | 16.4% | 24.4% | 65.3% |
| ALEXANDRIA | 45.3% | 56.7% | 78.9% |
| Oracle-Entity | 73.1% | 80.6% | 90.7% |

| Step | Δ | p |
|---|---|---|
| Base → Corpus-RAG  reach alone | -6.4 pp | 1.46e-02 |
| Corpus-RAG → Corpus-RAG + Entity  entity parsing | +9.9 pp | 8.80e-05 |
| Corpus-RAG + Entity → ALEXANDRIA  ALEXANDRIA architecture | +28.9 pp | 2.05e-11 |
| Base → ALEXANDRIA  full system | +32.4 pp | 3.62e-12 |

| System | Accuracy | Coverage | Acc when covered |
|---|---|---|---|
| Base | 4.8% | — | — |
| Index-RAG | 2.3% | 6.0% | 34.1% |
| Corpus-RAG | 9.1% | 13.1% | 60.4% |
| Corpus-RAG + Entity | 18.9% | 24.7% | 73.5% |
| ALEXANDRIA | 24.7% | 31.7% | 72.5% |
| Oracle-Entity | 40.5% | 52.5% | 76.9% |

| Step | Δ | p |
|---|---|---|
| Base → Corpus-RAG  reach alone | +4.3 pp | 3.78e-04 |
| Corpus-RAG → Corpus-RAG + Entity  entity parsing | +9.8 pp | 4.60e-16 |
| Corpus-RAG + Entity → ALEXANDRIA  ALEXANDRIA architecture | +5.8 pp | 6.98e-04 |
| Base → ALEXANDRIA  full system | +19.9 pp | 4.10e-31 |


### Majority-class baselines with oracle relation labels

| Relation | n | Distinct gold answers | Modal-guess accuracy | Base (1.7B, closed-book) | A − modal |
|---|---|---|---|---|---|
| country | 274 | 410 | 16.1% | 56.6% | +40.5 |
| author | 198 | 380 | 1.5% | 0.0% | -1.5 |
| sport | 196 | 68 | 71.9% | 52.6% | -19.4 |
| place of birth | 173 | 499 | 2.3% | 14.5% | +12.1 |
| occupation | 121 | 112 | 48.8% | 23.1% | -25.6 |
| genre | 113 | 191 | 11.5% | 7.1% | -4.4 |
| director | 95 | 124 | 3.2% | 0.0% | -3.2 |
| composer | 48 | 115 | 4.2% | 2.1% | -2.1 |
| screenwriter | 42 | 69 | 4.8% | 0.0% | -4.8 |
| religion | 40 | 6 | 72.5% | 17.5% | -55.0 |
| producer | 37 | 61 | 5.4% | 0.0% | -5.4 |
| capital | 29 | 34 | 3.4% | 65.5% | +62.1 |
| father | 27 | 75 | 7.4% | 3.7% | -3.7 |


### Relation mix, controlled

| Set (popularity matched) | n | Small-answer-space share | Base accuracy |
|---|---|---|---|
| Long-tail subset (all rare) | 1399 | 47.5% | 24.8% |
| Balanced sample, restricted to rare | 99 | 57.6% | 29.3% |

| Quantity | Value |
|---|---|
| Predicted for the rare-restricted sample from per-relation accuracies × its relation mix | 30.9% |
| Observed | 29.3% |
| **Residual after adjusting for relation mix** | **+1.6 pp** |


### Benchmark composition

| Relation | Share of 1a (long-tail) | Share of 1b (balanced) | Base accuracy on 1a |
|---|---|---|---|
| country | 19.6% | 5.4% | 56.6% |
| author | 14.2% | 11.5% | 0.0% |
| sport | 14.0% | 4.3% | 52.6% |
| place of birth | 12.4% | 3.3% | 14.5% |
| occupation | 8.6% | 2.9% | 23.1% |
| genre | 8.1% | 11.1% | 7.1% |
| director | 6.8% | 13.8% | 0.0% |
| composer | 3.4% | 7.0% | 2.1% |
| screenwriter | 3.0% | 12.6% | 0.0% |
| religion | 2.9% | 2.0% | 17.5% |
| producer | 2.6% | 12.0% | 0.0% |
| capital | 2.1% | 4.5% | 65.5% |
| father | 1.9% | 4.2% | 3.7% |
| capital of | 0.4% | 3.3% | — |
| mother | 0.1% | 1.7% | — |
| color | 0.0% | 0.4% | — |


## 3. Popularity crossover within a single dataset

| Decile | Median s_pop | Base | Corpus-RAG | ALEXANDRIA | Gemini 3.6 Flash | Corpus-RAG − A | B − Corpus-RAG | Corpus-RAG cov | B cov |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 30.0% | 24.0% | 49.0% | 54.0% | -6.0 | +25.0 | 21.0% | 52.0% |
| 2 | 136 | 16.0% | 19.0% | 32.0% | 61.0% | +3.0 | +13.0 | 27.0% | 37.0% |
| 3 | 237 | 12.0% | 18.0% | 23.0% | 62.0% | +6.0 | +5.0 | 27.0% | 36.0% |
| 4 | 423 | 8.0% | 28.0% | 33.0% | 71.0% | +20.0 | +5.0 | 30.0% | 41.0% |
| 5 | 734 | 8.0% | 27.0% | 31.0% | 75.0% | +19.0 | +4.0 | 36.0% | 40.0% |
| 6 | 1413 | 7.0% | 23.0% | 26.0% | 72.0% | +16.0 | +3.0 | 30.0% | 26.0% |
| 7 | 2467 | 3.0% | 34.0% | 32.0% | 73.0% | +31.0 | -2.0 | 39.0% | 42.0% |
| 8 | 5661 | 15.0% | 34.0% | 43.0% | 87.0% | +19.0 | +9.0 | 47.0% | 48.0% |
| 9 | 18039 | 12.0% | 32.0% | 37.0% | 88.0% | +20.0 | +5.0 | 36.0% | 47.0% |
| 10 | 76000 | 43.0% | 49.0% | 66.0% | 96.0% | +6.0 | +17.0 | 55.0% | 74.0% |

| Popularity stratum | Reach alone (Base → Corpus-RAG) | p | Architecture (Corpus-RAG → ALEXANDRIA) | p |
|---|---|---|---|---|
| Rarest half (deciles 1–5) | +8.4 pp | 2.64e-05 | +10.4 pp | 3.39e-06 |
| Most popular half (deciles 6–10) | +18.4 pp | 3.62e-16 | +6.4 pp | 5.93e-03 |


### PopQA long-tail (paired n = 1399)

| System | Accuracy | 95% CI | Coverage |
|---|---|---|---|
| Base | 24.8% | [22.7–27.0] | — |
| Index-RAG | 16.7% | [14.7–18.7] | 27.1% |
| Corpus-RAG | 24.4% | [22.2–26.6] | 22.9% |
| ALEXANDRIA | 45.6% | [43.0–48.2] | 49.5% |

| Isolates | Δ | Discordant | p |
|---|---|---|---|
| FULL-CORPUS RETRIEVAL vs NONE (Base → Corpus-RAG) | -0.4 pp | 110 / 104 | 7.33e-01 (ns) |
| ARCHITECTURE, reach held constant (Corpus-RAG → ALEXANDRIA) | +21.2 pp | 57 / 354 | 1.84e-53 |
| INDEX REACH, architecture ~constant (Index-RAG → Corpus-RAG) | +7.7 pp | 66 / 174 | 2.10e-12 |
| full system vs no retrieval (Base → ALEXANDRIA) | +20.8 pp | 70 / 361 | 2.56e-48 |


### core-400 (1a+1c) (paired n = 400)

| System | Accuracy | 95% CI | Coverage |
|---|---|---|---|
| Base | 25.2% | [21.0–29.5] | — |
| Index-RAG | 22.5% | [18.5–26.8] | 31.8% |
| Corpus-RAG | 32.8% | [28.2–37.5] | 33.2% |
| ALEXANDRIA | 52.2% | [47.2–57.2] | 58.5% |

| Isolates | Δ | Discordant | p |
|---|---|---|---|
| FULL-CORPUS RETRIEVAL vs NONE (Base → Corpus-RAG) | +7.5 pp | 27 / 57 | 1.40e-03 |
| ARCHITECTURE, reach held constant (Corpus-RAG → ALEXANDRIA) | +19.5 pp | 17 / 95 | 2.51e-14 |
| INDEX REACH, architecture ~constant (Index-RAG → Corpus-RAG) | +10.2 pp | 20 / 61 | 5.66e-06 |
| full system vs no retrieval (Base → ALEXANDRIA) | +27.0 pp | 14 / 122 | 1.10e-22 |


### NQ-Open (paired n = 500)

| System | Accuracy | 95% CI | Coverage |
|---|---|---|---|
| Base | 18.4% | [15.0–21.8] | — |
| Index-RAG | 33.0% | [29.0–37.2] | 45.4% |
| Corpus-RAG | 36.4% | [32.2–40.6] | 46.2% |
| ALEXANDRIA | 39.0% | [34.8–43.4] | 54.2% |

| Isolates | Δ | Discordant | p |
|---|---|---|---|
| FULL-CORPUS RETRIEVAL vs NONE (Base → Corpus-RAG) | +18.0 pp | 30 / 120 | 5.97e-14 |
| ARCHITECTURE, reach held constant (Corpus-RAG → ALEXANDRIA) | +2.6 pp | 50 / 63 | 2.59e-01 (ns) |
| INDEX REACH, architecture ~constant (Index-RAG → Corpus-RAG) | +3.4 pp | 53 / 70 | 1.49e-01 (ns) |
| full system vs no retrieval (Base → ALEXANDRIA) | +20.6 pp | 28 / 131 | 3.95e-17 |


## 4. Coverage decomposition

| Benchmark | System | Accuracy | Coverage | Acc when covered | Acc when NOT covered |
|---|---|---|---|---|---|
| 1a | raw Qwen3-1.7B (no retrieval) | 24.8% | — | — | — |
| 1a | pre-built-index RAG (no query-time corpus access) | 16.7% | 27.1% | 51.7% | 3.6% |
| 1a | full-corpus RAG — equal reach to B | 24.4% | 22.9% | 65.1% | 12.2% |
| 1a | dense-only (appendix) | 11.7% | 24.2% | 35.5% | 4.1% |
| 1a | BM25-only (appendix) | 12.5% | 17.4% | 53.1% | 4.0% |
| 1a | ALEXANDRIA (Pi) | 45.6% | 49.5% | 81.8% | 10.1% |
| 1b | raw Qwen3-1.7B (no retrieval) | 15.4% | — | — | — |
| 1b | pre-built-index RAG (no query-time corpus access) | 22.2% | 30.4% | 66.1% | 3.0% |
| 1b | full-corpus RAG — equal reach to B | 28.8% | 34.8% | 74.4% | 4.4% |
| 1b | ALEXANDRIA (Pi) | 37.2% | 44.3% | 79.0% | 3.9% |
| 1c | raw Qwen3-1.7B (no retrieval) | 18.0% | — | — | — |
| 1c | pre-built-index RAG (no query-time corpus access) | 22.4% | 30.2% | 64.2% | 4.3% |
| 1c | ALEXANDRIA (Pi) | 52.5% | 63.0% | 78.8% | 7.7% |
| 1d | raw Qwen3-1.7B (no retrieval) | 18.4% | — | — | — |
| 1d | pre-built-index RAG (no query-time corpus access) | 33.0% | 45.4% | 65.6% | 5.9% |
| 1d | full-corpus RAG — equal reach to B | 36.4% | 46.2% | 70.1% | 7.4% |
| 1d | BM25-only (appendix) | 23.4% | 31.4% | 63.1% | 5.2% |
| 1d | ALEXANDRIA (Pi) | 39.0% | 54.2% | 65.3% | 7.9% |


## 5. Matched-population test: effect of weak evidence

| Benchmark | n uncovered | A (no retrieval) | ALEXANDRIA | A − B | Discordant | p |
|---|---|---|---|---|---|---|
| PopQA long-tail | 706 | 11.5% | 10.1% | +1.4 | 33 / 23 | 2.29e-01 (ns) |
| PopQA popularity deciles | 557 | 5.0% | 3.9% | +1.1 | 18 / 12 | 3.62e-01 (ns) |
| EntityQuestions | 444 | 5.6% | 7.7% | -2.1 | 10 / 19 | 1.36e-01 (ns) |
| NQ-Open | 229 | 9.6% | 7.9% | +1.7 | 14 / 10 | 5.41e-01 (ns) |


### Grounding rate by evidence strength

| Benchmark | Grounded rate — retrieval MISSED | Grounded rate — retrieval HIT | Difference |
|---|---|---|---|
| PopQA long-tail | 77.1% | 91.8% | +14.7 pp |
| PopQA popularity deciles | 88.5% | 96.6% | +8.1 pp |
| EntityQuestions | 78.8% | 93.9% | +15.1 pp |
| NQ-Open | 87.3% | 98.5% | +11.2 pp |


## 6. Evidence gate: causal test

| System (paired n=1399) | Accuracy |
|---|---|
| ALEXANDRIA | 45.6% |
| ALEXANDRIA (no gate) | 44.5% |

| Condition | n | A (no retrieval) | B (gate on) | G (gate off) | Gate value (B − G) | 95% CI | Discordant | p |
|---|---|---|---|---|---|---|---|---|
| uncovered (retrieval missed) | 706 | 11.5% | 10.1% | 8.2% | +1.9 pp | [+0.3, +3.4] | 22 / 9 | 2.94e-02 |
| covered (retrieval hit) | 693 | 38.4% | 81.8% | 81.4% | +0.4 pp | [-2.0, +2.9] | 39 / 36 | 8.18e-01 (ns) |


## 7. Failure decomposition — ALEXANDRIA

| Benchmark | n | Accuracy | Retrieval failure | Generation failure |
|---|---|---|---|---|
| PopQA long-tail | 1399 | 45.6% | 50.5% | 9.0% |
| PopQA popularity deciles | 1000 | 37.2% | 55.7% | 9.3% |
| EntityQuestions | 1200 | 52.5% | 37.0% | 13.3% |
| NQ-Open | 500 | 39.0% | 45.8% | 18.8% |


### Accuracy by relation (1a, ALEXANDRIA)

| Relation | Accuracy | n |
|---|---|---|
| capital | 82.8% | 24/29 |
| country | 77.7% | 213/274 |
| sport | 66.3% | 130/196 |
| capital of | 60.0% | 3/5 |
| occupation | 59.5% | 72/121 |
| place of birth | 46.2% | 80/173 |
| father | 40.7% | 11/27 |
| religion | 37.5% | 15/40 |
| author | 22.2% | 44/198 |
| genre | 15.9% | 18/113 |
| composer | 14.6% | 7/48 |
| screenwriter | 14.3% | 6/42 |
| director | 12.6% | 12/95 |
| producer | 8.1% | 3/37 |
| mother | 0.0% | 0/1 |


## 8. Ablations (core-400, paired n = 400)

| Configuration | Accuracy | Δ | 95% CI on Δ | Coverage | Median latency | Discordant | Verdict |
|---|---|---|---|---|---|---|---|
| **FULL SYSTEM** | **52.2%** | — | — | 58.5% | 21.6 s | — | — |
| deep_archive = off (no query-time ZIM search) | 27.2% | -25.0 | [-29.5, -20.2] | 29.2% | 17.7 s | 116 | real effect |
| deep tier ONLY — all three fast-index generators off (core-400) | 51.0% | -1.2 | [-3.2, +0.8] | 56.5% | 19.9 s | 17 | underpowered |
| no multi-query deep search | 43.8% | -8.5 | [-12.2, -4.8] | 50.0% | 20.8 s | 58 | real effect |
| final_k = 1 | 48.2% | -4.0 | [-6.8, -1.2] | 47.2% | 13.6 s | 32 | real effect |
| final_k = 5 | 52.5% | +0.2 | [-1.8, +2.2] | 63.0% | 30.7 s | 15 | equivalent (within ±3 pp) |
| no deep-locate (lead chunks only) | 51.0% | -1.2 | [-3.0, +0.5] | 57.5% | 21.4 s | 13 | equivalent (within ±3 pp) |
| no FTS5 title generator | 52.2% | +0.0 | [-1.0, +1.0] | 58.5% | 21.0 s | 4 | equivalent (within ±3 pp) |
| no BM25 generator | 52.2% | +0.0 | [-1.5, +1.5] | 60.0% | 20.4 s | 10 | equivalent (within ±3 pp) |
| no dense-vector generator | 52.2% | +0.0 | [-1.2, +1.5] | 58.0% | 20.2 s | 8 | equivalent (within ±3 pp) |
| no entity-span title lookup | 52.2% | +0.0 | [-0.8, +0.8] | 58.5% | 22.4 s | 2 | equivalent (within ±3 pp) |
| epistemic fusion off | 52.2% | +0.0 | [+0.0, +0.0] | 58.5% | 22.4 s | 0 | identical output |
| ablation_sources = wiki_pure | 50.8% | -1.5 | [-2.8, -0.5] | 59.0% | 23.4 s | 6 | real effect |
| ablation_sources = wiki_deep | 52.2% | +0.0 | [+0.0, +0.0] | 58.5% | 24.5 s | 0 | identical output |


### Deep-tier-only, on the full headline set

| Configuration (paired n=1,399) | Accuracy | Coverage | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|
| ALEXANDRIA | 45.6% | 49.5% | — | — | — |
| Deep tier only — fast index disabled | 45.4% | 47.0% | -0.2 pp | [-1.4, +1.0] | EQUIVALENT within ±3 pp |


### Discrimination and operating point

| Benchmark | n | AUROC | 95% CI | Overall acc | Gate coverage (zone='grounded') | Gate acc | Gate lift |
|---|---|---|---|---|---|---|---|
| PopQA long-tail | 1399 | 0.645 | [0.616–0.675] | 45.6% | 84.3% | 48.1% | +2.5 |
| PopQA popularity deciles | 1000 | 0.716 | [0.684–0.749] | 37.2% | 92.1% | 38.7% | +1.5 |
| EntityQuestions | 1200 | 0.682 | [0.650–0.711] | 52.5% | 88.3% | 55.9% | +3.4 |
| NQ-Open | 500 | 0.628 | [0.577–0.675] | 39.0% | 93.4% | 40.3% | +1.3 |


### Discrimination vs random abstention at matched coverage

| Benchmark | Coverage | Threshold | Accuracy | vs random abstention |
|---|---|---|---|---|
| PopQA long-tail | 30% | 6.41 | 61.8% | +16.2 |
| PopQA long-tail | 50% | 5.29 | 56.1% | +10.5 |
| PopQA long-tail | 70% | 3.78 | 52.1% | +6.5 |
| PopQA popularity deciles | 30% | 6.80 | 60.0% | +22.8 |
| PopQA popularity deciles | 50% | 5.91 | 51.8% | +14.6 |
| PopQA popularity deciles | 70% | 4.65 | 46.0% | +8.8 |
| EntityQuestions | 30% | 7.09 | 70.6% | +18.1 |
| EntityQuestions | 50% | 5.78 | 65.8% | +13.3 |
| EntityQuestions | 70% | 4.33 | 60.7% | +8.2 |
| NQ-Open | 30% | 8.25 | 54.0% | +15.0 |
| NQ-Open | 50% | 7.14 | 44.8% | +5.8 |
| NQ-Open | 70% | 5.79 | 45.4% | +6.4 |


### Calibration — accuracy by reranker-logit band, pooled

| Logit range | n | Accuracy | 95% CI | Shipped zone |
|---|---|---|---|---|
| -99.0 to -4.0 | 86 | 37.2% | [26.7–47.7] | unverified |
| -4.0 to -2.0 | 76 | 28.9% | [19.7–39.5] | unverified |
| -2.0 to 0.0 | 91 | 20.9% | [13.2–29.7] | unverified |
| 0.0 to 0.5 | 34 | 32.4% | [17.6–47.1] | unverified |
| 0.5 to 2.0 | 183 | 25.1% | [19.1–31.7] | TENTATIVE |
| 2.0 to 4.0 | 587 | 28.1% | [24.5–31.7] | grounded |
| 4.0 to 6.0 | 1132 | 40.7% | [37.9–43.6] | grounded |
| 6.0 to 8.0 | 1349 | 51.3% | [48.6–53.9] | grounded |
| 8.0 to 99.0 | 561 | 69.0% | [65.1–72.7] | grounded |


### Zone-conditioned accuracy

| Benchmark | Zone | n | Share | Accuracy | 95% CI |
|---|---|---|---|---|---|
| PopQA long-tail | grounded | 1180 | 84.3% | 48.1% | [45.3–51.0] |
| PopQA long-tail | tentative | 77 | 5.5% | 19.5% | [11.7–28.6] |
| PopQA long-tail | unverified | 142 | 10.2% | 38.7% | [31.0–46.5] |
| PopQA popularity deciles | grounded | 921 | 92.1% | 38.7% | [35.5–41.8] |
| PopQA popularity deciles | tentative | 36 | 3.6% | 16.7% | [5.6–30.6] |
| PopQA popularity deciles | unverified | 43 | 4.3% | 23.3% | [11.6–34.9] |
| EntityQuestions | grounded | 1060 | 88.3% | 55.9% | [52.9–59.0] |
| EntityQuestions | tentative | 56 | 4.7% | 42.9% | [30.4–55.4] |
| EntityQuestions | unverified | 84 | 7.0% | 15.5% | [8.3–23.8] |
| NQ-Open | grounded | 467 | 93.4% | 40.3% | [35.8–44.8] |
| NQ-Open | tentative | 14 | 2.8% | 7.1% | [0.0–21.4] |
| NQ-Open | unverified | 18 | 3.6% | 33.3% | [11.1–55.6] |
| NQ-Open | direct | 1 | 0.2% | 0.0% | [0.0–0.0] |


## 9. Frontier model with our retrieved context

| Benchmark | Gemini 3.6 Flash | Gemini Flash + our context | Δ | p |
|---|---|---|---|---|
| PopQA long-tail | 58.8% | 59.4% | +0.6 | 6.26e-01 (ns) |
| PopQA popularity deciles | 73.9% | 62.0% | -11.9 | 5.43e-18 |
| EntityQuestions | 74.6% | 71.7% | -2.9 | 1.74e-02 |
| NQ-Open | 68.2% | 65.6% | -2.6 | 1.36e-01 (ns) |


### Local reader with evidence vs frontier closed-book

| Benchmark | n covered | B (1.7B, Pi) | Gemini 3.6 Flash (frontier, closed-book) | B − Gemini 3.6 Flash | Discordant | p |
|---|---|---|---|---|---|---|
| PopQA long-tail | 693 | 81.8% | 77.6% | +4.2 | 100 / 71 | 3.20e-02 |
| PopQA popularity deciles | 443 | 79.0% | 90.5% | -11.5 | 18 / 69 | 3.32e-08 |
| EntityQuestions | 756 | 78.8% | 88.0% | -9.2 | 60 / 129 | 5.73e-07 |
| NQ-Open | 271 | 65.3% | 80.8% | -15.5 | 20 / 62 | 3.71e-06 |


### Decomposed by whether the retrieved passages contained the answer

| Benchmark | Condition | n | Base | ALEXANDRIA | Gemini 3.6 Flash | Gemini Flash + our context | Gemini Flash + our context − Gemini 3.6 Flash |
|---|---|---|---|---|---|---|---|
| PopQA long-tail | covered | 693 | 38.4% | 81.8% | 77.6% | 88.7% | +11.1 |
| PopQA long-tail | uncovered | 706 | 11.5% | 10.1% | 40.2% | 30.6% | -9.6 |
| PopQA long-tail | *received passages* | 1257/1399 |  |  |  |  |  |
| PopQA popularity deciles | covered | 443 | 28.4% | 79.0% | 90.5% | 91.6% | +1.1 |
| PopQA popularity deciles | uncovered | 557 | 5.0% | 3.9% | 60.7% | 38.4% | -22.3 |
| PopQA popularity deciles | *received passages* | 957/1000 |  |  |  |  |  |
| EntityQuestions | covered | 756 | 25.3% | 78.8% | 88.0% | 93.3% | +5.3 |
| EntityQuestions | uncovered | 444 | 5.6% | 7.7% | 51.8% | 34.9% | -16.9 |
| EntityQuestions | *received passages* | 1116/1200 |  |  |  |  |  |
| NQ-Open | covered | 271 | 25.8% | 65.3% | 80.8% | 84.1% | +3.3 |
| NQ-Open | uncovered | 229 | 9.6% | 7.9% | 53.3% | 43.7% | -9.6 |
| NQ-Open | *received passages* | 481/500 |  |  |  |  |  |


## 10. Accuracy vs entity popularity, all arms (1b)

| Decile | Median s_pop | Base | Index-RAG | ALEXANDRIA | Gemini 3.6 Flash | B − Index-RAG | B − Gemini 3.6 Flash | B cov | Index-RAG cov |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 30.0% | 27.0% | 49.0% | 54.0% | +22.0 | -5.0 | 52.0% | 34.0% |
| 2 | 136 | 16.0% | 9.0% | 32.0% | 61.0% | +23.0 | -29.0 | 37.0% | 17.0% |
| 3 | 237 | 12.0% | 10.0% | 23.0% | 62.0% | +13.0 | -39.0 | 36.0% | 18.0% |
| 4 | 423 | 8.0% | 9.0% | 33.0% | 71.0% | +24.0 | -38.0 | 41.0% | 12.0% |
| 5 | 734 | 8.0% | 8.0% | 31.0% | 75.0% | +23.0 | -44.0 | 40.0% | 14.0% |
| 6 | 1413 | 7.0% | 12.0% | 26.0% | 72.0% | +14.0 | -46.0 | 26.0% | 17.0% |
| 7 | 2467 | 3.0% | 18.0% | 32.0% | 73.0% | +14.0 | -41.0 | 42.0% | 22.0% |
| 8 | 5661 | 15.0% | 29.0% | 43.0% | 87.0% | +14.0 | -44.0 | 48.0% | 39.0% |
| 9 | 18039 | 12.0% | 36.0% | 37.0% | 88.0% | +1.0 | -51.0 | 47.0% | 52.0% |
| 10 | 76000 | 43.0% | 64.0% | 66.0% | 96.0% | +2.0 | -30.0 | 74.0% | 79.0% |


## 11. Judge overlay and inter-judge agreement

| Benchmark | Decisions | Judge–judge agreement | Cohen's κ | Gemini positive rate | GPT positive rate | Judge–containment agreement |
|---|---|---|---|---|---|---|
| PopQA long-tail | 1900 | 97.2% | 0.943 | 48.4% | 48.9% | 92.5% |
| PopQA popularity deciles | 2064 | 96.7% | 0.927 | 35.4% | 36.3% | 92.8% |
| EntityQuestions | 2076 | 96.9% | 0.938 | 45.9% | 46.1% | 91.6% |
| NQ-Open | 2090 | 93.0% | 0.859 | 50.7% | 53.8% | 84.0% |


### Per system

| Benchmark | System | Containment | Judge-G | Judge-O | Δ G | Δ O | Committed | G–O agreement |
|---|---|---|---|---|---|---|---|---|
| PopQA long-tail | Base | 31.3% | 31.1% | 29.7% | -0.3 | -1.6 | 91.1% | 98.2% |
| PopQA long-tail | ALEXANDRIA | 55.8% | 59.5% | 58.7% | +3.7 | +2.9 | 93.9% | 98.2% |
| PopQA long-tail | Gemini 3.6 Flash | 71.3% | 67.6% | 69.7% | -3.7 | -1.6 | 78.7% | 95.3% |
| PopQA long-tail | Gemini Flash + our context | 69.2% | 69.5% | 71.6% | +0.3 | +2.4 | 81.6% | 96.3% |
| PopQA long-tail | Index-RAG | 18.4% | 14.2% | 14.7% | -4.2 | -3.7 | 55.8% | 97.9% |
| PopQA popularity deciles | Base | 12.8% | 12.8% | 12.0% | +0.0 | -0.8 | 89.9% | 98.4% |
| PopQA popularity deciles | ALEXANDRIA | 38.8% | 39.5% | 39.5% | +0.8 | +0.8 | 93.4% | 98.4% |
| PopQA popularity deciles | Gemini 3.6 Flash | 82.6% | 69.2% | 74.6% | -13.4 | -7.9 | 75.8% | 90.7% |
| PopQA popularity deciles | Index-RAG | 20.7% | 20.0% | 19.0% | -0.8 | -1.7 | 71.5% | 99.0% |
| EntityQuestions | Base | 12.9% | 17.7% | 15.8% | +4.8 | +2.9 | 95.4% | 97.7% |
| EntityQuestions | ALEXANDRIA | 55.9% | 63.0% | 64.0% | +7.1 | +8.1 | 93.4% | 96.7% |
| EntityQuestions | Gemini 3.6 Flash | 80.3% | 80.9% | 83.6% | +0.6 | +3.3 | 87.9% | 95.0% |
| EntityQuestions | Index-RAG | 20.0% | 21.8% | 21.2% | +1.7 | +1.2 | 63.4% | 98.3% |
| NQ-Open | Base | 18.2% | 18.7% | 20.1% | +0.5 | +1.9 | 85.2% | 95.7% |
| NQ-Open | ALEXANDRIA | 41.9% | 45.7% | 48.3% | +3.8 | +6.5 | 90.4% | 94.0% |
| NQ-Open | Gemini 3.6 Flash | 74.6% | 78.2% | 83.3% | +3.6 | +8.6 | 90.9% | 89.7% |
| NQ-Open | Gemini Flash + our context | 72.0% | 74.9% | 80.4% | +2.9 | +8.4 | 92.6% | 91.6% |
| NQ-Open | Index-RAG | 34.4% | 36.1% | 37.1% | +1.7 | +2.6 | 81.8% | 93.8% |


### Three-way metric validation

| Outcome (n = 8,130 decisions) | Share |
|---|---|
| All three metrics agree | 88.4% |
| Judges agree with each other, matcher differs | 7.5% |
| — of which matcher too **strict** (missed a correct answer) | 4.5% |
| — of which matcher too **loose** (false hit) | 3.0% |
| Judges disagree with each other | 4.1% |
| **Net matcher bias** | **1.5 pp conservative** |


## 12. Systems measurements

|  | Pooled (n=4099) | Controlled run (n=100) |
|---|---|---|
| median | 22.6 s | 23.5 s |
| p90 | 28.9 s | 28.1 s |
| p99 | 37.0 s | 34.6 s |
| max | 76.4 s | 34.6 s |
| min | 4.1 s | 8.9 s |
| mean | 22.8 s | — |
| throughput | 158 queries/hour | — |


### Query budget (median, n=4091 non-deliberate)

| Stage | Time | Share |
|---|---|---|
| Retrieval | 4.61 s | 20.4% |
| **Prefill of retrieved context** | **15.37 s** | **68.1%** |
| Decode | 2.58 s | 11.4% |
| **Wall** | **22.56 s** |  |


### Latency by zone

| Zone | n | Median latency |
|---|---|---|
| grounded | 3628 | 22.9 s |
| unverified | 287 | 10.3 s |
| tentative | 183 | 23.7 s |
| direct | 1 | 4.4 s |


### Reproducibility (n=100, identical questions)

| System | run 1 | run 2 | Verdict agreement | Byte-identical |
|---|---|---|---|---|
| **ALEXANDRIA (temp 0.2)** | 60.0% | 60.0% | 100.0% | 100.0% |
| **Gemini 3.6 Flash (temp 0)** | 65.0% | 66.0% | 93.0% | 20.0% |


### Answer length and abstention (1a)

| System | Median words | p90 | Abstain / hedge |
|---|---|---|---|
| Base | 9 | 33 | 1.1% |
| Index-RAG | 17 | 59 | 16.2% |
| Corpus-RAG | 12 | 55 | 13.5% |
| ALEXANDRIA | 10 | 40 | 1.5% |
| Gemini 3.6 Flash | 12 | 55 | 14.8% |
| Gemini Flash + our context | 9 | 38 | 13.0% |
| Gemini 3.1 Pro | 9 | 55 | 26.3% |


## 13. Data integrity

| File | Rows | Empty/unscored | Errors | Kind |
|---|---|---|---|---|
| 1a_A | 1399 | 0 | 0 | generation |
| 1a_B | 1399 | 0 | 0 | generation |
| 1a_C1 | 1399 | 0 | 0 | generation |
| 1a_C2 | 1399 | 0 | 0 | generation |
| 1a_G | 1399 | 0 | 0 | generation |
| 1a_N3 | 78 | 0 | 0 | generation |
| 1a_N4 | 1399 | 0 | 0 | generation |
| 1a_N5 | 1399 | 0 | 0 | generation |
| 1a_N5e | 1399 | 0 | 0 | generation |
| 1a_N5p | 1399 | 0 | 0 | generation |
| 1a_N_bm25 | 1399 | 0 | 0 | generation |
| 1a_N_dense | 1399 | 0 | 0 | generation |
| 1a_OA_gpt-4_1-mini | 1399 | 0 | 0 | exploratory (excluded) |
| 1a_OA_gpt-4o-2024-05-13 | 1399 | 0 | 0 | exploratory (excluded) |
| 1a_OA_gpt-4o | 1399 | 0 | 0 | exploratory (excluded) |
| 1a_P2COL | 1399 | 0 | 0 | generation |
| 1a_judge | 1900 | 0 | 0 | judge |
| 1a_judge2 | 1900 | 0 | 0 | judge |
| 1b_A | 1000 | 0 | 0 | generation |
| 1b_B | 1000 | 0 | 0 | generation |
| 1b_C1 | 1000 | 0 | 0 | generation |
| 1b_C2 | 1000 | 0 | 0 | generation |
| 1b_N4 | 1000 | 0 | 0 | generation |
| 1b_N5 | 1000 | 0 | 0 | generation |
| 1b_P2COL | 1000 | 0 | 0 | generation |
| 1b_judge | 2064 | 0 | 0 | judge |
| 1b_judge2 | 2064 | 0 | 0 | judge |
| 1c_A | 1200 | 0 | 0 | generation |
| 1c_B | 1200 | 0 | 0 | generation |
| 1c_C1 | 1200 | 0 | 0 | generation |
| 1c_C2 | 1200 | 0 | 0 | generation |
| 1c_N4 | 1200 | 0 | 0 | generation |
| 1c_P2COL | 1200 | 0 | 0 | generation |
| 1c_judge | 2076 | 0 | 0 | judge |
| 1c_judge2 | 2076 | 0 | 0 | judge |
| 1d_A | 500 | 0 | 0 | generation |
| 1d_B | 500 | 0 | 0 | generation |
| 1d_C1 | 500 | 0 | 0 | generation |
| 1d_C2 | 500 | 0 | 0 | generation |
| 1d_N4 | 500 | 0 | 0 | generation |
| 1d_N5 | 500 | 0 | 0 | generation |
| 1d_N_bm25 | 500 | 0 | 0 | generation |
| 1d_P2COL | 500 | 0 | 0 | generation |
| 1d_judge | 2090 | 0 | 0 | judge |
| 1d_judge2 | 2090 | 0 | 0 | judge |
| ablation baseline | 400 | 0 | 0 | generation |
| deep tier only | 400 | 0 | 0 | generation |
| deep tier only | 1399 | 0 | 0 | generation |
| final_k = 1 | 400 | 0 | 0 | generation |
| final_k = 5 | 400 | 0 | 0 | generation |
| no bm25 | 400 | 0 | 0 | generation |
| no deep | 400 | 0 | 0 | generation |
| no entity | 400 | 0 | 0 | generation |
| no fusion | 400 | 0 | 0 | generation |
| no evidence gate | 400 | 53 | 53 | generation |
| no locate | 400 | 0 | 0 | generation |
| no multiq | 400 | 0 | 0 | generation |
| no title | 400 | 0 | 0 | generation |
| no vector | 400 | 0 | 0 | generation |
| Wikipedia deep only | 400 | 0 | 0 | generation |
| Wikipedia only | 400 | 0 | 0 | generation |
| core400_A | 400 | 0 | 0 | generation |
| core400_N4 | 400 | 0 | 0 | generation |
| core400_N5 | 400 | 0 | 0 | generation |
| lat_controlled | 100 | 0 | 0 | generation |
| pilot_A | 150 | 0 | 0 | generation |
| pilot_B | 150 | 0 | 0 | generation |
| pilot_C1 | 150 | 0 | 0 | generation |
| pilot_C2 | 150 | 0 | 0 | generation |
| var_B_run1 | 100 | 0 | 0 | generation |
| var_B_run2 | 100 | 0 | 0 | generation |
| var_C1_run1 | 100 | 0 | 0 | generation |
| var_C1_run2 | 100 | 0 | 0 | generation |
