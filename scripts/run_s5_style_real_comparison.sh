#!/bin/bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eventssm

export JAX_VISIBLE_DEVICES=${JAX_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}

MODES=("independent_real_decay" "real_rotation2x2")
SEEDS=(1234 2345 3456 4567 5678)
MODEL_CONFIG=${MODEL_CONFIG:-shd/s5_hw_tiny}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/shd_s5_style_real_comparison_v1}
EPOCHS=${EPOCHS:-30}

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="${OUTPUT_ROOT}/${mode}/seed_${seed}"
    echo "Running model=${MODEL_CONFIG} a_mode=${mode} seed=${seed}"

    python run_training.py \
      task=spiking-heidelberg-digits \
      model="${MODEL_CONFIG}" \
      seed="${seed}" \
      training.num_epochs="${EPOCHS}" \
      training.num_workers=0 \
      training.validate_on_test=false \
      model.ssm.a_mode="${mode}" \
      output_dir="${outdir}" \
      logging.interval=1000
  done
done
