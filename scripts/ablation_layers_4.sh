#!/bin/bash
#SBATCH --job-name=ucr_tdi_small
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eventssm

export JAX_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

MODES=("shared_real_decay" "independent_real_decay" "real_rotation2x2")
SEEDS=(1234 2345 3456)
LAYERS=(4)

for layers in "${LAYERS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      outdir="outputs/ablation_layers/${layers}_layers/${mode}/seed_${seed}"

      python run_training.py \
        task=tutorial \
        seed=${seed} \
        training.num_epochs=20 \
        training.num_workers=0 \
        model.ssm.a_mode=${mode} \
        model.ssm.num_layers_per_stage=${layers} \
        model.ssm.dropout=0.0 \
        training.drop_event=0.0 \
        training.cut_mix=0.0 \
        training.noise=0 \
        training.time_jitter=0 \
        training.spatial_jitter=0 \
        training.time_skew=1.0 \
        output_dir=${outdir} \
        logging.interval=1000
    done
  done
done