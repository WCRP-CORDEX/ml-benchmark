#!/bin/bash
#SBATCH --job-name=cordex-plots
#SBATCH --output=logs/cordex-plots-MULTI-%j.out
#SBATCH --error=logs/cordex-plots-MULTI-%j.err
#SBATCH --time=14:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cordex-ml-benchmark

# ============================================================================
#  CORDEX-ML-Bench :: full benchmarking pipeline
#  ----------------------------------------------
#  PART 0  build_scorecard.py
#  PART 1  Daily snapshot plots          (primary GCM)
#  PART 2  Historical climatology plots  (primary GCM)
#  PART 3  Climate-change signal plots   (both GCMs per domain)
#  PART 4  Combined region plots         (ALPS + SA + NZ stacked)
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
DATE="${DATE:-}"
SEED="${SEED:-42}"
MEMBER="${MEMBER:-}"
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
# 4) COMBINED REGIONS (ALPS + SA + NZ stacked)  —  climatology + daily
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 4 - Combined region plots (climatology + daily)"
echo "============================================================"

for var in "${VARIABLES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do

    echo; echo ">>> combined  $var / $exp / climatology"
    pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
        --variable "$var" \
        --experiment "$exp" \
        --mode climatology \
        --truth-root "$TRUTH_ROOT" \
        --pred-root  "$PRED_ROOT" \
        --out-dir    "$OUT_ROOT/combined" \
        --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
        --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"

    echo; echo ">>> combined  $var / $exp / daily"
    pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
        --variable "$var" \
        --experiment "$exp" \
        --mode daily \
        --truth-root "$TRUTH_ROOT" \
        --pred-root  "$PRED_ROOT" \
        --out-dir    "$OUT_ROOT/combined" \
        --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
        --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"

  done
done

# ---------------------------------------------------------------------------
# 4b) COMBINED REGIONS — climate-change signal
# ---------------------------------------------------------------------------
echo; echo "============================================================"
echo "  PART 4b - Combined region cc_signal plots"
echo "============================================================"

for var in "${VARIABLES[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    for future in mid_century end_century; do

      echo; echo ">>> combined  $var / $exp / cc_signal / $future"
      pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
          --variable "$var" \
          --experiment "$exp" \
          --mode cc_signal \
          --future-period "$future" \
          --truth-root "$TRUTH_ROOT" \
          --pred-root  "$PRED_ROOT" \
          --out-dir    "$OUT_ROOT/combined" \
          --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
          --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"

      # pr only: also produce the relative (%) version
      if [[ "$var" == "pr" ]]; then
        echo; echo ">>> combined  $var / $exp / cc_signal / $future (relative %)"
        pixi run python "$SCRIPTS_DIR/plot_combined_regions.py" \
            --variable "$var" \
            --experiment "$exp" \
            --mode cc_signal \
            --future-period "$future" \
            --relative \
            --truth-root "$TRUTH_ROOT" \
            --pred-root  "$PRED_ROOT" \
            --out-dir    "$OUT_ROOT/combined" \
            --seed "$SEED" --pool "$POOL" --n-sample "$N_SAMPLE" \
            --bot-pool "$BOT_POOL" --bot-exclude "$BOT_EXCLUDE"
      fi

    done
  done
done

# ---------------------------------------------------------------------------
# 3) CLIMATE CHANGE SIGNAL  (both GCMs per domain, single-domain plots)
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
echo "   combined/      — stacked ALPS+SA+NZ figures (climatology / daily / cc_signal)"
echo "   cc_signal/     — per-domain climate-change signal plots"