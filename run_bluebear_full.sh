#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sam3-full-trial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:0
#SBATCH --output=logs/full-trial-%j.out

# Runs the ENTIRE pipeline (decode + segment + postprocess) in a
# single GPU job, for convenience while testing. Not the recommended
# production pattern - CPU-bound stages (decode, postprocess) sit idle on
# an A100 the whole time, wasting scarce GPU allocation. Use run_bluebear.sh
# (the three-job split) for actual production runs.

set -e

CONFIG="${1:-config.json}"

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source sam3-env/bin/activate

echo "Using config: $CONFIG"
echo

bash run_local.sh --config "$CONFIG"