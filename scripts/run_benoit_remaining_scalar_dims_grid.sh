#!/usr/bin/env bash
# Resume the scalar per-dimension Manifesto DSPy ladders leaf-size by leaf-size
# without rerunning the completed economic ladder or completed decentralization
# leaves.

set -euo pipefail

SCALAR_ROOT="${1:-outputs/manifesto_fg_alternating/scalar_dims_benoit_g0init_fresh_dspy_20260427_001903}"
DECENTRALIZATION_ROOT="${DECENTRALIZATION_ROOT:-outputs/manifesto_fg_alternating/decentralization_benoit_g0init_fresh_dspy_20260426_1815}"
DIMENSIONS="${DIMENSIONS:-social immigration eu environment}"
DECENTRALIZATION_LEAF_FILTER="${DECENTRALIZATION_LEAF_FILTER:-4096,8192}"
RUN_DECENTRALIZATION="${RUN_DECENTRALIZATION:-1}"

common_env=(
  MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
  FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-g}"
  INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
  INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-0}"
  STAGE_NAMING="${STAGE_NAMING:-powers}"
  DSPY_BUDGET="${DSPY_BUDGET:-medium}"
  DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}"
  DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}"
  DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}"
  DSPY_MIPRO_NUM_CANDIDATES="${DSPY_MIPRO_NUM_CANDIDATES:-2}"
  DSPY_MIPRO_NUM_TRIALS="${DSPY_MIPRO_NUM_TRIALS:-6}"
  DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS="${DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS:-0}"
  DSPY_MIPRO_MAX_LABELED_DEMOS="${DSPY_MIPRO_MAX_LABELED_DEMOS:-0}"
  DSPY_MIPRO_MINIBATCH_SIZE="${DSPY_MIPRO_MINIBATCH_SIZE:-128}"
  DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-3}"
  KEEP_LAST_SERVER="${KEEP_LAST_SERVER:-0}"
  PLOT_PREDICTION_DISTS="${PLOT_PREDICTION_DISTS:-0}"
)

run_dimensions="${DIMENSIONS}"
dimension_root_overrides="${DIMENSION_ROOT_OVERRIDES:-}"
dimension_leaf_filters="${DIMENSION_LEAF_FILTERS:-}"
if [[ "${RUN_DECENTRALIZATION}" == "1" ]]; then
  case " ${run_dimensions} " in
    *" decentralization "*) ;;
    *) run_dimensions="${run_dimensions} decentralization" ;;
  esac
  dimension_root_overrides="${dimension_root_overrides} decentralization=${DECENTRALIZATION_ROOT}"
  dimension_leaf_filters="${dimension_leaf_filters} decentralization=${DECENTRALIZATION_LEAF_FILTER}"
fi

echo "=== $(date -u) :: scalar remaining dimensions root=${SCALAR_ROOT} dims=${run_dimensions} ==="
env "${common_env[@]}" \
  DIMENSIONS="${run_dimensions}" \
  DIMENSION_ROOT_OVERRIDES="${dimension_root_overrides}" \
  DIMENSION_LEAF_FILTERS="${dimension_leaf_filters}" \
  bash scripts/run_benoit_multi_dimension_ladder_context_groups.sh "${SCALAR_ROOT}"

echo "=== $(date -u) :: scalar remaining dimensions resume done ==="
