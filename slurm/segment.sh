#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sam3-segment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:0
#SBATCH --output=logs/segment-%j.out

set -e

CONFIG="${1:-config.json}"

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate

echo "== segment_frame =="
python src/segment_frame.py --config "$CONFIG"

echo "== merge_masks =="
python src/merge_masks.py --config "$CONFIG"