#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sam3-joint-grid-search
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=1:00:0
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/projects/d/dilucam-arme/sam3-background-removal/logs/joint-grid-search-%j.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=talbotm@bham.ac.uk

set -e

# Usage:
#   sbatch slurm/joint_grid_search.sh --manifest gt_manifest.json --grid-config joint_grid_config.json --output results.csv
#   sbatch slurm/joint_grid_search.sh --frame f.png --gt-mask gt.png --grid-config joint_grid_config.json --output results.csv
#
# All args after the script name are passed straight through to joint_grid_search.py.
# Bump --time above if your grid is large - Slurm kills the job at the limit
# regardless of progress (though CSV rows written so far are kept, since
# results are written row-by-row as the search runs, not all at the end).

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal/grid_search
export HF_HUB_CACHE=/rds/projects/d/dilucam-arme/sam3-background-removal/hf_cache

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

source /rds/projects/d/dilucam-arme/sam3-background-removal/sam3-env/bin/activate
cd ${SAM3_PROJECT_ROOT}


echo "== joint_grid_search =="
python src/joint_grid_search.py "$@"
