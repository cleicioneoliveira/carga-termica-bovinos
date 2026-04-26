#!/usr/bin/env bash
# ==============================================================================
# run_pipeline_1293.sh
#
# Executes the full farm 1293 processing chain:
#   1. environment correction
#   2. health-status timeline reconstruction
#   3. monitoring + health merge
#   4. thermal-load/comfort analysis
#
# Run from the repository root:
#   bash scripts/run_pipeline_1293.sh
#
# Requirements:
#   - environment_correction must be importable in the active Python environment.
#   - status_timeline_reconstructor must be importable.
#   - merge_monitoramento_saude must be importable.
#   - this repository must be the current working directory for the final stage.
# ============================================================================

set -Eeuo pipefail

RAW_DIR="${RAW_DIR:-dataset/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-dataset/processado}"
REPORTS_DIR="${REPORTS_DIR:-${PROCESSED_DIR}/reports}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

HEAT_FILE="${HEAT_FILE:-${RAW_DIR}/heat_stress_report_f1293.csv}"
MONITORAMENTO_RAW="${MONITORAMENTO_RAW:-${RAW_DIR}/monitoramento_full.csv}"
SAUDE_RAW="${SAUDE_RAW:-${RAW_DIR}/saude_1293.xlsx}"

MONITORAMENTO_CORRIGIDO="${MONITORAMENTO_CORRIGIDO:-${PROCESSED_DIR}/monitoramento_corrigido.csv}"
SAUDE_TIMELINE="${SAUDE_TIMELINE:-${PROCESSED_DIR}/saude_timeline_final.parquet}"
DATASET_FINAL="${DATASET_FINAL:-${PROCESSED_DIR}/monitoramento_saude_unificado.parquet}"

CONFIG_FILE="${CONFIG_FILE:-app/config.yaml}"
SHOW_PLOTS="${SHOW_PLOTS:-0}"

info() {
  printf '[INFO] %s\n' "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf '[ERROR] Required file not found: %s\n' "$path" >&2
    exit 1
  fi
}

mkdir -p "$PROCESSED_DIR" "$REPORTS_DIR"

info "Checking raw inputs."
require_file "$HEAT_FILE"
require_file "$MONITORAMENTO_RAW"
require_file "$SAUDE_RAW"

info "Step 1/4: correcting environmental variables."
python -m environment_correction.environment_correction \
  --heat "$HEAT_FILE" \
  --monitoramento "$MONITORAMENTO_RAW" \
  --output-monitoramento "$MONITORAMENTO_CORRIGIDO" \
  --output-audit "$REPORTS_DIR/device_lag_audit.csv" \
  --output-pairs "$REPORTS_DIR/device_pair_candidates.csv" \
  --output-summary "$REPORTS_DIR/correction_summary.json" \
  --output-quality "$REPORTS_DIR/quality_summary.csv" \
  --output-inconsistencies "$REPORTS_DIR/environment_inconsistencies.csv" \
  --output-coverage "$REPORTS_DIR/correction_coverage.csv" \
  --lag-min -6 \
  --lag-max 6 \
  --min-overlap-hours 72 \
  --lag-mode shared \
  --humidity-unit auto \
  --aggregation mean \
  --min-score-margin 0.05 \
  --log-level "$LOG_LEVEL"

require_file "$MONITORAMENTO_CORRIGIDO"

info "Step 2/4: reconstructing health-status timeline."
python -m status_timeline_reconstructor.status_timeline_reconstructor \
  --input "$SAUDE_RAW" \
  --output-dir "$PROCESSED_DIR" \
  --id-col brinco \
  --datetime-col data_mudanca_status \
  --status-col status_saude \
  --previous-status-col status_saude_anterior \
  --next-status-col prox_status_saude \
  --output-datetime-col data_hora \
  --analysis-start "2025-01-01 00:00:00" \
  --analysis-end "2025-12-31 23:00:00" \
  --freq h \
  --valid-status Desafio \
  --valid-status Observacao \
  --valid-status Grave \
  --valid-status Normal \
  --status-alias "observação=Observacao" \
  --status-alias "observacao=Observacao" \
  --status-alias "obs=Observacao" \
  --conflict-policy flag \
  --normalize-columns \
  --prefix saude \
  --log-level "$LOG_LEVEL"

require_file "$SAUDE_TIMELINE"

info "Step 3/4: merging corrected monitoring with health timeline."
python -m merge_monitoramento_saude.cli \
  --monitoramento "$MONITORAMENTO_CORRIGIDO" \
  --saude "$SAUDE_TIMELINE" \
  --output-dir "$PROCESSED_DIR"

require_file "$DATASET_FINAL"

info "Step 4/4: running thermal-load and comfort analysis."
if [[ "$SHOW_PLOTS" == "1" ]]; then
  python -m app.run_pipeline --config "$CONFIG_FILE" --show-plots --log-level "$LOG_LEVEL"
else
  python -m app.run_pipeline --config "$CONFIG_FILE" --log-level "$LOG_LEVEL"
fi

info "Integrated pipeline completed successfully."
info "Final dataset: $DATASET_FINAL"
info "Thermal outputs: outputs_conforto/"
