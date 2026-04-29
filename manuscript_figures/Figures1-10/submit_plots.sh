#!/bin/bash
#SBATCH --job-name=cordex-plots
#SBATCH --output=logs/cordex-plots-%j.out
#SBATCH --error=logs/cordex-plots-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark

# ============================================================================
#  CORDEX-ML-Bench :: full benchmarking pipeline
#  ----------------------------------------------
#  PART 0  build_scorecard.py
#            - walks diagnostics/*/<DOMAIN>_Domain/<experiment>/<variable>/
#              <config>/metric_*.nc
#            - produces  per-domain (ALPS / NZ / SA) + averaged scorecards
#            - writes    model_rankings.json  (consumed by PART 1–3)
#  PART 1  Daily snapshot plots          (primary GCM)
#  PART 2  Historical climatology plots  (primary GCM)
#  PART 3  Climate-change signal plots   (both GCMs per domain)
# ============================================================================

set -euo pipefail
mkdir -p logs
echo "=== Job started on $(hostname) at $(date) ==="

# --- pixi ---
export PATH="$HOME/.local/bin:$PATH"
export PIXI_HOME=/lustre/gmeteo/WORK/rampaln/.pixi
export PIXI_CACHE_DIR=/lustre/gmeteo/WORK/rampaln/.pixi/cache

PIXI_PROJECT=/lustre/gmeteo/WORK/rampaln/cordex-ml-bench
cd "$PIXI_PROJECT"
echo "PIXI_HOME=$PIXI_HOME"
pixi info

pixi add python=3.11 xarray dask netcdf4 zarr matplotlib numpy cartopy pandas >/dev/null

# --- script directory (where this file lives) ---
SCRIPTS_DIR="/lustre/gmeteo/WORK/rampaln/cordex-ml-bench"

# --- user-configurable paths ---
DIAG_ROOT="${DIAG_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/diagnostics}"
TRUTH_ROOT="${TRUTH_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/target}"
PRED_ROOT="${PRED_ROOT:-/lustre/gmeteo/WORK/abadj/cordex-bench-eval/data/staging}"
OUT_ROOT="${OUT_ROOT:-/lustre/gmeteo/WORK/rampaln/cordex-ml-bench/plots}"

# --- knobs for the plot scripts ---
DATE="${DATE:-}"                # e.g. 1995-07-14 ; blank = auto (wettest/hottest)
SEED="${SEED:-42}"
MEMBER="${MEMBER:-}"            # blank = ensemble mean
POOL="${POOL:-10}"
N_SAMPLE="${N_SAMPLE:-3}"
BOT_POOL="${BOT_POOL:-15}"
BOT_EXCLUDE="${BOT_EXCLUDE:-3}"
DAILY_MODE="${DAILY_MODE:-auto}"

declare -A PRIMARY_GCM=(
    [ALPS]=CNRM-CM5
    [NZ]=ACCESS-CM2
    [SA]=ACCESS-CM2
)
declare -A CC_GCMS=(
    [ALPS]="CNRM-CM5 MPI-ESM-LR"
    [NZ]="ACCESS-CM2 EC-Earth3"
    [SA]="ACCESS-CM2 NorESM2-MM"
)

DOMAINS=(ALPS NZ SA)
VARIABLES=(pr tasmax)
EXPERIMENTS=(ESD_pseudo_reality Emulator_hist_future)

cd "$PIXI_PROJECT"

# ---------------------------------------------------------------------------
# 0) SCORECARD  (also writes model_rankings.json used by steps 1–3)
# ---------------------------------------------------------------------------
# echo; echo "============================================================"
# echo "  PART 0 - Scorecards + rankings from diagnostics/"
# echo "============================================================"
# pixi run python "$SCRIPTS_DIR/build_scorecard.py" \
#     --diagnostics-root "$DIAG_ROOT" \
#     --out-dir "$OUT_ROOT/scorecard" \
#     --rankings-out "$SCRIPTS_DIR/model_rankings.json"

# ---------------------------------------------------------------------------
# 1) DAILY SNAPSHOTS  (primary GCM)
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 1 - Daily snapshot plots"
echo "============================================================"
for dom in "${DOMAINS[@]}"; do
  gcm="${PRIMARY_GCM[$dom]}"
  for var in "${VARIABLES[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
      echo; echo ">>> daily  $dom / $var / $exp / $gcm"
      extra=()
      if [[ "$DAILY_MODE" == "date" && -n "$DATE" ]]; then
        extra+=(--mode date --date "$DATE")
      else
        extra+=(--mode "$DAILY_MODE")
      fi
      [[ -n "$SEED"   ]] && extra+=(--seed "$SEED")
      [[ -n "$MEMBER" ]] && extra+=(--member "$MEMBER")
      pixi run python "$SCRIPTS_DIR/plot_daily_snapshot.py" \
          --domain "$dom" --gcm "$gcm" --variable "$var" \
          --experiment "$exp" --period historical \
          --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
          --out-dir "$OUT_ROOT/daily" \
          --pool "$POOL" --n-sample "$N_SAMPLE" \
          --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE" \
          "${extra[@]}"
      # two extra random days are done also
      pixi run python "$SCRIPTS_DIR/plot_daily_snapshot.py" \
          --domain "$dom" --gcm "$gcm" --variable "$var" \
          --experiment "$exp" --period historical \
          --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
          --out-dir "$OUT_ROOT/daily" \
          --pool "$POOL" --n-sample "$N_SAMPLE" \
          --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE" --mode "random"
      pixi run python "$SCRIPTS_DIR/plot_daily_snapshot.py" \
          --domain "$dom" --gcm "$gcm" --variable "$var" \
          --experiment "$exp" --period historical \
          --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
          --out-dir "$OUT_ROOT/daily" \
          --pool "$POOL" --n-sample "$N_SAMPLE" \
          --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE" --mode "random"
    done
  done
done

# ---------------------------------------------------------------------------
# 2) CLIMATOLOGIES
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 2 - Historical climatology plots"
echo "============================================================"
for dom in "${DOMAINS[@]}"; do
  gcm="${PRIMARY_GCM[$dom]}"
  for var in "${VARIABLES[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
      echo; echo ">>> clim  $dom / $var / $exp / $gcm"
      pixi run python "$SCRIPTS_DIR/plot_climatology.py" \
          --domain "$dom" --gcm "$gcm" --variable "$var" \
          --experiment "$exp" --period historical \
          --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
          --out-dir "$OUT_ROOT/climatology" \
          --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
          --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"
    done
  done
done

# ---------------------------------------------------------------------------
# 4) COMBINED REGIONS (ALPS + SA + NZ stacked)
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 4 - Combined region plots"
echo "============================================================"

for var in "${VARIABLES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do

    echo; echo ">>> combined  $var / $exp / climatology"
    pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
        --variable "$var" \
        --experiment "$exp" \
        --mode climatology \
        --truth-root "$TRUTH_ROOT" \
        --pred-root "$PRED_ROOT" \
        --out-dir "$OUT_ROOT/combined" \
        --seed "$SEED" \
        --pool "$POOL" \
        --n-sample "$N_SAMPLE" \
        --bot-pool "$BOT_POOL" \
        --bot-exclude "$BOT_EXCLUDE"

    echo; echo ">>> combined  $var / $exp / daily"
    pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
        --variable "$var" \
        --experiment "$exp" \
        --mode daily \
        --truth-root "$TRUTH_ROOT" \
        --pred-root "$PRED_ROOT" \
        --out-dir "$OUT_ROOT/combined" \
        --seed "$SEED" \
        --pool "$POOL" \
        --n-sample "$N_SAMPLE" \
        --bot-pool "$BOT_POOL" \
        --bot-exclude "$BOT_EXCLUDE"

  done
done

# ---------------------------------------------------------------------------
# 3) CLIMATE CHANGE SIGNAL  (both GCMs per domain)
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 3 - Climate change signal plots (both GCMs per domain)"
echo "============================================================"
for dom in "${DOMAINS[@]}"; do
  for gcm in ${CC_GCMS[$dom]}; do
    for var in "${VARIABLES[@]}"; do
      for exp in "${EXPERIMENTS[@]}"; do
        for future in mid_century end_century; do
          echo; echo ">>> ccs  $dom / $var / $exp / $gcm / $future"
          pixi run python "$SCRIPTS_DIR/plot_climate_change.py" \
              --domain "$dom" --gcm "$gcm" --variable "$var" \
              --experiment "$exp" --future-period "$future" \
              --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
              --out-dir "$OUT_ROOT/cc_signal" \
              --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
              --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"
          if [[ "$var" == "pr" ]]; then
            pixi run python "$SCRIPTS_DIR/plot_climate_change.py" \
                --domain "$dom" --gcm "$gcm" --variable "$var" \
                --experiment "$exp" --future-period "$future" \
                --truth-root "$TRUTH_ROOT" --pred-root "$PRED_ROOT" \
                --out-dir "$OUT_ROOT/cc_signal" \
                --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
                --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE" \
                --relative
          fi
        done
      done
    done
  done
done

echo; echo "=== Job finished at $(date) ==="
echo "Outputs under: $OUT_ROOT"
echo "   scorecard/     — 8 scorecard figures + raw_scores.csv"
echo "   daily/         — daily-snapshot plots"
echo "   climatology/   — Rx1day / TXx climatologies"
echo "   cc_signal/     — climate-change signal plots + KDE"
