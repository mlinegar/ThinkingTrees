#!/usr/bin/env bash
# 4-way econ ablation, one per GPU, traces skipped (TT_SKIP_FULL_TREE_TRACES=1).
set -u
cd /home/mlinegar/ThinkingTrees
export TT_SKIP_FULL_TREE_TRACES=1
B="--family fno --embedding-backend local-hf --embedding-model /mnt/data/models/google/embeddinggemma-300m --embedding-device cuda --embedding-batch-size 64 --leaf-qsentences 16 --max-iterations 2 --fno-epochs 8 --fno-batch-size 16 --fno-learning-rate 3e-3 --fno-target-dimension economic"

run() { # gpu grid modes hc nl hh lw mw rw out
  local gpu=$1 grid=$2 modes=$3 hc=$4 nl=$5 hh=$6 lw=$7 mw=$8 rw=$9 out=${10}
  rm -rf "outputs/$out"
  CUDA_VISIBLE_DEVICES=$gpu ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py $B \
    --fg-grid-dir "$grid" --fno-n-modes "$modes" --fno-hidden-channels "$hc" --fno-n-layers "$nl" --fno-head-hidden-dim "$hh" \
    --fno-leaf-weight "$lw" --fno-merge-weight "$mw" --fno-root-weight "$rw" \
    --output-dir "outputs/$out" > "outputs/$out.log" 2>&1 &
  echo "[ablation] GPU$gpu -> $out (pid $!)"
}

run 0 outputs/benoit_chunkgrid_forced_economic_llmspan  768 128 4 256 1.0 1.0 10.0 econ_llmspan_bigG
run 1 outputs/benoit_chunkgrid_forced_economic_meanroll 768 128 4 256 1.0 1.0 10.0 econ_meanroll_bigG
run 2 outputs/benoit_chunkgrid_forced_economic_llmspan  384  64 3 128 1.0 1.0 10.0 econ_llmspan_modG
run 3 outputs/benoit_chunkgrid_forced_economic_llmspan  768 128 4 256 1.0 0.5 30.0 econ_llmspan_bigG_rootheavy
echo "[ablation] all 4 launched; waiting"
wait
echo "[ablation] all done"
