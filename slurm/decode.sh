#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbdefault
#SBATCH --job-name=sam3-decode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:0
#SBATCH --output=logs/decode-%j.out

set -e

CONFIG="${1:-config.json}"

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate

echo "== setup_dirs =="
python src/setup_dirs.py --config "$CONFIG"

echo "== decode_video =="
python src/decode_video.py --config "$CONFIG"