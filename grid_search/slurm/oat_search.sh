#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sam3-oat-search
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=0:30:0
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/projects/d/dilucam-arme/sam3-background-removal/logs/oat-search-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=talbotm@bham.ac.uk

set -e

# Usage:
#   sbatch slurm/oat_search.sh --manifest gt_manifest.json --oat-config oat_config_wide.json --output oat_wide_results.csv
#   sbatch slurm/oat_search.sh --frame f.png --gt-mask gt.png --oat-config oat_config_wide.json --output oat_wide_results.csv
#
# All args after the script name are passed straight through to oat_search.py.
# This is a small, fast run (well under 200 SAM3 calls for the wide config) -
# 30 min is generous headroom, not a real estimate of how long it'll take.

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal/grid_search
export HF_HUB_CACHE=/rds/projects/d/dilucam-arme/sam3-background-removal/hf_cache

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

source /rds/projects/d/dilucam-arme/sam3-background-removal/sam3-env/bin/activate
cd ${SAM3_PROJECT_ROOT}

echo "== oat_search =="
python -u src/oat_search.py "$@"