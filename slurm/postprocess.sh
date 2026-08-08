#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbdefault
#SBATCH --job-name=sam3-postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:0
#SBATCH --output=logs/postprocess-%j.out

set -e

CONFIG="${1:-config.json}"

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal

module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate

echo "== remove_fg_noise =="
python src/remove_fg_noise.py --config "$CONFIG"

echo "== fill_small_holes =="
python src/fill_small_holes.py --config "$CONFIG"

echo "== opening =="
python src/opening.py --config "$CONFIG"

echo "== erosion =="
python src/erosion.py --config "$CONFIG"

echo "== smooth_gaussian =="
python src/smooth_gaussian.py --config "$CONFIG"

echo "== composite_and_stitch =="
python src/composite_and_stitch.py --config "$CONFIG"

echo "== cleanup_tmp =="
python src/cleanup_tmp.py --config "$CONFIG"