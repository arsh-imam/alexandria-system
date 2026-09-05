#!/bin/bash
# Class I ablations: in-process config overrides only. Engine files untouched.
# Runs on core400 (200 PopQA-LT + 200 EQ, seed 20260702, frozen).
cd "$(dirname "$0")"
[ -n "${VIRTUAL_ENV:-}" ] || echo "note: activate your venv first"
set -x
# I1 corpus scope
python3 run_arm_b.py --set core400.jsonl --out abl_wiki_pure  --tag I1a --override ablation_sources=wiki_pure
python3 run_arm_b.py --set core400.jsonl --out abl_wiki_deep  --tag I1b --override ablation_sources=wiki_deep
# I2 epistemic fusion off
python3 run_arm_b.py --set core400.jsonl --out abl_nofusion   --tag I2  --override use_epistemic_fusion=False
# I4 leave-one-generator-out
python3 run_arm_b.py --set core400.jsonl --out abl_notitle    --tag I4a --override use_title=False
python3 run_arm_b.py --set core400.jsonl --out abl_nobm25     --tag I4b --override use_bm25=False
python3 run_arm_b.py --set core400.jsonl --out abl_novector   --tag I4c --override use_vector=False
python3 run_arm_b.py --set core400.jsonl --out abl_nodeep     --tag I4d --override deep_archive=off
# I5 k sensitivity
python3 run_arm_b.py --set core400.jsonl --out abl_k1         --tag I5a --override final_k=1
python3 run_arm_b.py --set core400.jsonl --out abl_k5         --tag I5b --override final_k=5
# deep-tier internals
python3 run_arm_b.py --set core400.jsonl --out abl_nolocate   --tag I6a --override deep_locate=False
python3 run_arm_b.py --set core400.jsonl --out abl_nomultiq   --tag I6b --override deep_multiquery=False
python3 run_arm_b.py --set core400.jsonl --out abl_noentity   --tag I6c --override entity_titles=False
# baseline at same config for paired comparison
python3 run_arm_b.py --set core400.jsonl --out abl_baseline   --tag I0
