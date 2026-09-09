# ALEXANDRIA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22314038.svg)](https://doi.org/10.5281/zenodo.22314038)

A fully-offline retrieval-augmented assistant: **Qwen3-1.7B (Q4_K_M) on a
Raspberry Pi 5, 8 GB, CPU-only**. It stores **41 ZIM archives totalling
82.12 GB**; the query-time archive registry exposes **35 of them, 65.30 GB**, to
deep retrieval — English Wikipedia always, plus 34 specialist archives routed per
query. The remaining six contributed passages to the **2,028,337-passage**
pre-built index but are not opened by the deep retriever. Normal operation has no
network dependency after provisioning.

This repository is the **frozen system as evaluated**. The twelve modules in
`src/` are byte-identical to the ones under test; `manifests/code_hashes.tsv`
records their SHA-256 and gate G0 verifies them.

## Install

Any Linux host, x86-64 or aarch64, with **120 GB free**:

    git clone https://github.com/arsh-imam/alexandria-system
    cd alexandria-system
    python3 install.py /path/to/data
    python3 src/main.py

`install.py` downloads the 96.66 GB of artifacts from the companion dataset at
https://huggingface.co/datasets/arshimam/alexandria-system, verifies all 53
objects by SHA-256, reassembles the split Wikipedia archive and verifies the
result, exposes the tree where the frozen modules expect it, and runs the
verification gates. Nothing is built and no source code is patched.

The download is resumable: if it stops, re-run the same command and completed
files are skipped.

Install to a directory that does not already contain a copy of the system.

## Verification

    python3 verify/verify_alexandria.py

| gate | checks | needs |
|---|---|---|
| G0 | 12 modules byte-identical to manifest | clone |
| G1 | pinned dependency versions | clone |
| G2 | query/index embedding alignment, self_cos >= 0.98 | clone + 133 MB embedder |
| G3 | 160-word windows, 40-word overlap, per-source caps | clone |
| G4 | index cardinality and ID-layout integrity | index |
| G5 | aarch64 CPU_REPACK kernels active | GGUF, aarch64 only |
| G6 | 41 stored archives by size and ZIM UUID; SHA-256 with `--deep` | corpus |
| G7 | fixed query set reproduces retrieval exactly | full install |
| G8 | decode byte-identical at 0.2 / 0.4 / 0.6 | GGUF, aarch64 |

G0 through G3 need no corpus. G0 and G3 read only files in the clone; G1 and G2
additionally need the pinned Python environment installed, and G2 needs the
embedding model — which it fetches on first use, at the exact revision the index
was built with, verifying it by hash. Together they take about five minutes.

Gates state their own prerequisites and **skip rather than fail** when those are
absent; a skipped gate is reported as unverified, never as passed.

**G2 is the load-bearing one.** It embeds 1,000 sampled passages through the
real `AlignedEmbedder` and scores them against their stored vectors, alongside
two controls: a random-pair floor showing the measurement can produce low
values, and a deliberate re-run of the historically mismatched embedding path
that must fail the threshold. Measured on the evaluated system: aligned
1.000000, random-pair floor 0.4094, mismatched path 0.8648.

## Architecture-dependent claims

Reference values for **G5** and **G8** were recorded on aarch64. llama.cpp
selects different SIMD kernels per architecture. Decode is **not** byte-identical
across architectures: the same prompts at temperatures 0.2, 0.4 and 0.6 produced
different output on x86-64 than the aarch64 reference in every case. Both gates
therefore skip on a non-aarch64 host and say why. The latency figures reported in
the paper are Raspberry Pi 5 measurements.

Retrieval identity across architectures has not been tested: that would require
G7 run with the full corpus on both hosts, which has not been done. What was
measured on both aarch64 (Raspberry Pi 5) and x86-64 (Debian under WSL 2):

| | result |
|---|---|
| Embedding alignment (G2) | identical, worst 1.000000 on both |
| Index integrity (G4) | identical, 2,028,337 after a Hugging Face round trip |
| Artifact hashes | all objects verified by SHA-256 on both |
| Decode (G8) | **differs** |

The embedding path is not architecture-dependent. G2 was run on both aarch64
(Raspberry Pi 5) and x86-64 (Debian under WSL 2) against the same 1,000-passage
fixture and returned identical values to six decimal places on both: worst
1.000000, mean 1.000000, random-pair floor 0.4094. Full retrieval identity
across architectures has not been tested, which would require G7 with the
complete corpus on both hosts.

The mismatch control inside G2 needs `fastembed`, which is a development
dependency and is not in `requirements-frozen.txt`. Without it the control
reports as unavailable and the gate still passes on its primary measurement.

## Layout

| path | contents |
|---|---|
| `src/` | the 12 frozen runtime modules |
| `build/` | the index build pipeline, verbatim as run |
| `verify/` | 9-gate verification suite and its fixtures |
| `manifests/` | hashes, environment locks, corpus and ZIM provenance |
| `install.py` | reviewer installer |
| `tools/setup_paths.sh` | bind-mount helper, used when /media is not writable |
| `evaluation/` | every per-query record, the statistics, and `RESULTS.md` |

## Evaluation

`evaluation/RESULTS.md` holds every reported figure, generated from 44,765
per-query records rather than written by hand. `evaluation/README.md` describes
what can be re-checked and how, from reading the tables to re-running individual
generations through the installed system.

## Notes on reproduction

The runtime modules use absolute paths under `/media/pi/KINGSTON/local_ai`.
`install.py` recreates that path with a symlink, or a bind mount where `/media`
is not writable, rather than editing the frozen source, which is what keeps G0
meaningful.

`build/` is published as it ran, including the Windows drive letters of the
build machines. Those paths are provenance: `build_titleset.py` and
`extract_passages.py` ran on one laptop (`G:`), `embed_gpu.py` on a second with
a GTX 1650 (`F:`), and `build_annoy.py`, `build_sqlite.py` and `build_fts.py` on
the Pi. `manifests/env_build.txt` records the exact interpreter and package set
of each.

`build_annoy.py` does not seed Annoy's tree construction, so `passages.ann` is
not byte-reproducible; the evaluated artifact is distributed and hash-pinned
instead. Every other stage is deterministic: chunking, FTS5 and BM25, the
cross-encoder, routing and the evidence gate.

`manifests/requirements-system.txt` lists the seven packages the twelve modules
import. `requirements-frozen.txt` is the full transitive closure at the exact
evaluated versions. `env_development.txt` is the development machine's complete
freeze and includes packages the system does not use.

## Licence

Code: see `LICENSE`. Distributed archives and models retain their upstream
licences, recorded on the dataset card at
https://huggingface.co/datasets/arshimam/alexandria-system
