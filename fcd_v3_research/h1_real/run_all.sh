#!/usr/bin/env bash
# Full H1 real-data run in the pre-registered order. Reads only ai_router/data/100k and ai_router/experiments.
set -euo pipefail
cd "$(dirname "$0")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
E="expert_1 expert_2 expert_3"
log "P1 metadata";                       python p1_metadata.py > /dev/null
log "P3.0 class map (Meta-Train)";       python p3_class_map.py
for w in meta val; do for e in $E; do
  if [ -f cache/pred_${w}_${e}.npz ]; then log "inference $w $e: cached"; continue; fi
  log "inference $w $e (no labels)";     python infer_cache.py $w $e | grep -E "^done"
done; done
for e in $E; do log "grid counts meta $e"; python build_features.py meta $e; done
for e in $E; do log "P3 pilot + registration $e"; python p3_pilot.py $e | grep -E "sha256|^  \"t_cls_min|theta_IoU|last_below|first_above"; done
log "--- registration frozen; Val labels are read from here on ---"
for e in $E; do log "grid counts val $e";  python build_features.py val $e; done
for e in $E; do log "Exp B $e";            python exp_b.py $e > results/log_exp_b_$e.txt; done
for e in $E; do log "Exp C' $e";           python exp_c_prime.py $e > results/log_exp_c_prime_$e.txt; done
for e in $E; do log "D11 ablation (coarse grid) $e"; python ablation_coarse_grid.py $e; done
for f in p1_report.json class_map.json; do cp cache/$f results/; done
for e in $E; do cp cache/registration_$e.json cache/p3_report_$e.json results/; done
log "self-tests";                        python test_h1core.py
log "ALLDONE"
