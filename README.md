# SAM3 Background Removal for ARME

Pipeline for removing backgrounds from musician performance footage using Meta's SAM3 model. Runs in two environments: locally, and on BlueBEAR HPC (CUDA/A100) for full runs.

## Example Directory Structure

```
Input_Videos/
└── IMG_5097.mov

Output_Videos/
└── IMG_5097.mov

tmp/
└── <video_name>/                       # e.g. tmp/IMG_5097/
    ├── decoded_frames/                  # raw frames from decode_video.py
    ├── segmented_frames/
    │   ├── person/                      # per-frame masks, prompt="person"
    │   ├── violin/                      # per-frame masks, prompt="violin"
    │   └── violin_bow/                  # per-frame masks, prompt="violin_bow"
    ├── merged_frames/                   # OR-merged masks, pre-postprocessing
    └── postprocessed_frames/            # after morphological postprocessing
```

---

## Setup

SAM3 is a gated model. Access needs to be configured for both environments, local vs BlueBEAR. After this step pick the section for whichever machine you're setting up.

### Gated SAM3 model access (required for both environments)

`facebook/sam3` is a gated model. Two separate steps are required (access request + local auth)

1. **Request access** on the model page: [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3). Your hf profile needs to be setup, or the "Request access" button won't appear at all.
2. **Authenticate on the machine you're running on**:
   ```bash
   hf auth login
   ```
   Paste a read-access token when prompted.


   Verify it worked:
   ```bash
   hf auth whoami
   ```

   If you hit stale-token issues:
   ```bash
   hf auth login --force
   ```

Do this separately on **each** machine you use (local Mac and BlueBEAR both need their own login).

---

### Local Setup

**1. Create the conda environment:**
```bash
conda create -n sam3-local python=3.11
conda activate sam3-local
pip install -r requirements.txt
```
(or alternatively use venv)

**2. Authenticate with Hugging Face** - see [Gated SAM3 model access](#gated-sam3-model-access-required-for-both-environments) above.

**3. Verify MPS is available (if on macOS):**
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
Should print `True`.

See [Running the Pipeline](#running-the-pipeline) below.

---

### BlueBEAR Setup (HPC)

**1. Request project access**

You'll need access to the `dilucam-arme` project to reach `/rds/projects/d/dilucam-arme`. Also confirm your account has access to the `bbgpu` (A100) partition via the BEAR Portal - segmentation jobs will fail without it.

If connecting off-campus (and sometimes on macOS even on-campus), you'll also need the remote access portal: https://remoteaccess.bham.ac.uk. This applies regardless of which option below you use.

**2. Connect - choose one:**

*Option A: BEAR Portal (browser-based, no local setup needed)*
https://portal.bear.bham.ac.uk - HPC Shell Access can be used directly from here.

*Option B: SSH*
```bash
ssh YOUR_USERNAME@bluebear.bham.ac.uk
```

**(Optional) Skip the password prompt on future SSH connections:**

```bash
ssh-keygen -t ed25519 -C "your.email@bham.ac.uk"   # default save location is fine
ssh-copy-id YOUR_USERNAME@bluebear.bham.ac.uk       # asks for your password one last time
ssh YOUR_USERNAME@bluebear.bham.ac.uk               # should now connect without a prompt
```
**3. Get the repo**

This should already be cloned at `/rds/projects/d/dilucam-arme/sam3-background-removal` — you likely just need to `cd` there and `git pull` rather than clone fresh:
```bash
cd /rds/projects/d/dilucam-arme/sam3-background-removal
git pull
```

If it isn't there yet:
```bash
cd /rds/projects/d/dilucam-arme
git clone git@github.com:arme-project/sam3-background-removal.git
```

Typical workflow is to edit locally and push, then `git pull` on BlueBEAR 

**4. Load required modules and set paths**

Run this at the start of **every fresh SSH session**, before creating or activating the virtual environment. Skipping this causes `pip install` to target the wrong Python environment, leading to confusing "missing package" errors later.
```bash
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HOME=${SAM3_PROJECT_ROOT}/hf_cache
```
*(This is usually handled automatically inside the Slurm scripts — you only need to run it manually if working interactively on the cluster.)*

**5. Create the virtual environment**

This should already be set up — you likely just need to activate it:
```bash
cd sam3-background-removal
source sam3-env/bin/activate
```

If it doesn't exist yet (or the project directory has been moved — venvs can't be relocated and must be rebuilt from scratch):
```bash
python -m venv sam3-env
source sam3-env/bin/activate
pip install -r requirements.txt
```

**6. Authenticate with Hugging Face** — see [Gated SAM3 model access](#gated-sam3-model-access-required-for-both-environments) above.

You're ready to run the pipeline — see [Running the Pipeline](#running-the-pipeline) below.

---

## Running the Pipeline

Works the same way on both environments once setup is complete. All scripts read from `config.json` in the current directory by default - pass `--config path/to/other.json` to override.

```bash
python src/setup_dirs.py
python src/decode_video.py
python src/segment_frame.py
```

On BlueBEAR, these are typically chained together and submitted via Slurm rather than run interactively — see `submit_pipeline.sh`.