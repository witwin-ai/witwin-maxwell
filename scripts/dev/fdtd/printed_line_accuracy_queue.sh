#!/usr/bin/env bash
# FDTD mesh / padding / material-separation convergence queue for the public
# printed CPW and coupled-pair columns. Prepared, NOT launched by its author.
#
# One column at a time on one GPU, nice 10, 8 threads, waits while the 1-min
# load exceeds LOAD_MAX, stops at the first failure, never retries, never
# overwrites: a finished column (execution.json present) is skipped, any other
# existing folder stops the queue so the partial/failed output is preserved.
#
#   STAGES="1 2"   (default) 36 columns + 36 Cu-only columns, ~12 GPU-h
#   STAGES="1 2 3" adds the finest transverse grid (12 columns), ~+13 GPU-h
#   DRY_RUN=1      print the plan only
#
# Usage: GPU=0 STAGES="1 2" scripts/dev/fdtd/printed_line_accuracy_queue.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PY:-python}  # the maxwell environment interpreter
OUT=${OUT:-${WITWIN_EVIDENCE_ROOT:-evidence}/pcb-fdtd-artifacts-20261029-accuracy}
GPU=${GPU:-0}
STAGES=${STAGES:-"1 2"}
LOAD_MAX=${LOAD_MAX:-200}
DRY_RUN=${DRY_RUN:-0}
LOG="$OUT/queue.log"

# label | overrides | CPW min/col | pair min/col | CPW GB | pair GB
# Estimates scale the measured A6000 columns (CPW 6.7 min / 5.2 GB, pair
# 5.8 min / 2.6 GB at dx 1.25 mm) by transverse cells x CFL steps; memory is
# scaled by cells (an upper bound). Shared-host timing, not a performance claim.
CONFIGS_1=(
  "T0_dx2.5|--dx 0.0025|3.0|2.6|4.0|2.0"
  "T2_dx0.625|--dx 0.000625|19.4|16.8|7.8|3.9"
  "P2_padding0.16|--padding 0.16|14.2|12.3|11.0|5.5"
)
CONFIGS_2=(
  "T0_dx2.5_Cu_only|--dx 0.0025 --sigma 0|3.0|2.6|4.0|2.0"
  "T1_dx1.25_Cu_only|--sigma 0|6.7|5.8|5.2|2.6"
  "T2_dx0.625_Cu_only|--dx 0.000625 --sigma 0|19.4|16.8|7.8|3.9"
)
CONFIGS_3=(
  "T3_dx0.3125|--dx 0.0003125|72|62|14.7|7.3"
)

mkdir -p "$OUT"
say() { echo "$*" | tee -a "$LOG"; }

wait_for_load() {
  while :; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l <= m)}'; then return; fi
    say "wait load $load > $LOAD_MAX $(date)"
    sleep 120
  done
}

wait_for_gpu() {  # need GB free on the selected device
  local need_mib=$(( ${1%.*} * 1024 + 2048 ))
  while :; do
    free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "$free" -ge "$need_mib" ]; then return; fi
    say "wait GPU$GPU free ${free}MiB < ${need_mib}MiB $(date)"
    sleep 120
  done
}

run_column() {  # kind length drive overrides minutes gb label
  local kind=$1 length=$2 drive=$3 overrides=$4 minutes=$5 gb=$6 label=$7
  local dir="$OUT/$label/$kind-L$length-D$drive"
  if [ -f "$dir/execution.json" ]; then say "skip finished $label $kind L$length D$drive"; return; fi
  if [ -e "$dir" ]; then say "STOP: partial output preserved at $dir"; exit 1; fi
  local cap=$(( ${minutes%.*} * 3 * 60 + 600 ))
  if [ "$DRY_RUN" = 1 ]; then say "plan $label $kind L$length D$drive ($overrides) ~${minutes} min ${gb} GB cap ${cap}s"; return; fi
  wait_for_load; wait_for_gpu "$gb"
  mkdir -p "$OUT/$label"
  say "launch $label $kind L$length D$drive $(date) load $(cut -d' ' -f1-3 /proc/loadavg)"
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 timeout "$cap" nice -n 10 \
      /usr/bin/time -v "$PY" -m scripts.dev.fdtd.printed_line_loss_execution --public-line "$kind" \
      --length-index "$length" --drive-index "$drive" $overrides --execute --output "$dir" ) \
      > "$OUT/$label/$kind-L$length-D$drive.log" 2>&1 \
    || { say "STOP: column failed $label $kind L$length D$drive (no retry)"; exit 1; }
  say "done $label $kind L$length D$drive $(date)"
}

reduce_config() {  # kind overrides label drives
  local kind=$1 overrides=$2 label=$3 drives=$4 short=pair
  [ "$kind" = cpw ] && short=cpw
  [ "$DRY_RUN" = 1 ] && { say "plan reduce $label $kind"; return; }
  [ -f "$OUT/$label/$short-two-length.json" ] && { say "skip reduced $label $kind"; return; }
  for length in 0 1; do
    local cols=()
    for drive in $(seq 0 $((drives - 1))); do cols+=("$OUT/$label/$kind-L$length-D$drive"); done
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 "$PY" -m scripts.dev.fdtd.printed_line_loss_bridge \
        basis --kind "$kind" --length-index "$length" $overrides --columns "${cols[@]}" \
        --output "$OUT/$label/$short-L$length-basis.json" ) > "$OUT/$label/reduce-$short-L$length.log" 2>&1 \
      || { say "STOP: basis reduction failed $label $kind L$length"; exit 1; }
  done
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 "$PY" -m scripts.dev.fdtd.printed_line_loss_bridge \
      two-length --kind "$kind" $overrides --short "$OUT/$label/$short-L0-basis.json" \
      --long "$OUT/$label/$short-L1-basis.json" --output "$OUT/$label/$short-two-length.json" ) \
      > "$OUT/$label/reduce-$short-two-length.log" 2>&1 \
    || { say "STOP: two-length reduction failed $label $kind"; exit 1; }
  say "reduced $label $kind $(date)"
}

say "queue start $(date) repo $REPO commit $(git -C "$REPO" rev-parse HEAD) GPU $GPU stages $STAGES"
for stage in $STAGES; do
  declare -n configs="CONFIGS_$stage"
  for entry in "${configs[@]}"; do
    IFS='|' read -r label overrides cpw_min pair_min cpw_gb pair_gb <<< "$entry"
    for kind in cpw differential_pair; do
      if [ "$kind" = cpw ]; then drives=2 minutes=$cpw_min gb=$cpw_gb; else drives=4 minutes=$pair_min gb=$pair_gb; fi
      for length in 0 1; do
        for drive in $(seq 0 $((drives - 1))); do
          run_column "$kind" "$length" "$drive" "$overrides" "$minutes" "$gb" "$label"
        done
      done
      reduce_config "$kind" "$overrides" "$label" "$drives"
    done
  done
done
say "QUEUE COMPLETE $(date)"
