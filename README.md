# SAM3 Background Removal for ARME

This is the pipeline for removing backgrounds specifically from musician performance footage (using Meta's SAM3 model). This runs in two environments: locally, and on BlueBEAR HPC (CUDA/A100).

## Contents

1. [Example Directory Structure](#1-example-directory-structure)
2. [Setup](#2-setup)
3. [Local Setup](#3-local-setup)
4. [BlueBEAR Setup (HPC)](#4-bluebear-setup-hpc)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Selecting Parameters / Editing Configuration Files](#6-selecting-parameters--editing-configuration-files)
7. [Troubleshooting](#7-troubleshooting)



## 1. Example Directory Structure

The main directory on the RDS is currently located in:
`/rds/projects/d/dilucam-arme/sam3-background-removal`

To navigate there from your home directory you will typically use: 
`cd /rds/projects/d/dilucam-arme/sam3-background-removal`

The repo follows this general structure:
```
Input_Videos/                        # Videos to process. Each filename (without extension) must match video_name in config.json
└── IMG_5097.mov

Output_Videos/                       # Processed videos are output here.
└── IMG_5097.mov

tmp/                                 # Temporary folder which stores intermediate frames between stages
└── <video_name>/                    # e.g. tmp/IMG_5097
    ├── decoded_frames/              # raw frames from decode_video.py — this writes out all video frames individually
    ├── segmented_frames/
    │   ├── person/                  # per-frame masks segmented using prompt="person"
    │   ├── violin/                  # per-frame masks segmented using prompt="violin"
    │   └── violin_bow/              # per-frame masks segmented using prompt="violin_bow"
    ├── merged_frames/               # OR-merged masks (merges the three individual masks), before postprocessing
    └── postprocessed_frames/        # after morphological postprocessing

src/                                 # There is one Python script per pipeline stage
hf_cache/                            # Downloaded SAM3 model weights (Hugging Face cache) - DO NOT DELETE
logs/                                # Slurm output files and stats files which may be helpful for debugging
sam3-env/                            # Python virtual environment on BlueBEAR. Activate with: source sam3-env/bin/activate

slurm/                               # Job scripts submitted by run_bluebear.sh
├── decode.sh                        # CPU job
├── segment.sh                       # A100 GPU job
└── postprocess.sh                   # CPU job

config.json                          # Settings for one run
pyrightconfig.json                   # Editor settings for the Pyright type checker. Not used by the pipeline
requirements.txt                     # Python packages, installed with: pip install -r requirements.txt
run_bluebear.sh                      # Launcher: submits the three jobs in slurm/ as a dependent chain. Run with bash, not sbatch
run_bluebear_full.sh                 # Whole pipeline in one Slurm job (1 hour limit). Short test clips only. Submit with sbatch
run_local.sh                         # Runs every stage in order on your own machine. Use --from-stage to resume partway
```



## 2. Setup

SAM3 is a gated model so you need to set up access - this is required for both local and BlueBEAR setup.

### Gated SAM3 model access (required for both environments)

`facebook/sam3` is a gated model. Two separate steps are required (access request + local auth)

1. **Request access** on the model page: [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3). Your hf profile needs to be set up, or the "Request access" button won't appear at all.
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

Do this separately on **each** machine you use (local and BlueBEAR both need their own login).



## 3. Local Setup

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

See [Running the Pipeline](#5-running-the-pipeline) below.



## 4. BlueBEAR Setup (HPC)

**1. Request project access**

You'll need access to the `dilucam-arme` project to reach `/rds/projects/d/dilucam-arme`. Also confirm your account has access to the `bbgpu` (A100) partition via the BEAR Portal - segmentation jobs will fail without it.

If connecting off-campus (and sometimes on macOS even on-campus), you'll also need the remote access portal: https://remoteaccess.bham.ac.uk. This applies regardless of which option below you use.

**2. Connect - choose one:**

*Option A: BEAR Portal (browser-based, no local setup needed)*
https://portal.bear.bham.ac.uk - HPC Shell Access can be used directly from here.
This is accessed by selecting the Clusters dropdown menu and selecting BlueBEAR HPC Shell Access.

*Option B: SSH*
```bash
ssh YOUR_USERNAME@bluebear.bham.ac.uk
```

For example mine is: ssh talbotm@bluebear.bham.ac.uk
(Yours should follow a similar structure, and should hopefully be provided when you are granted access to BlueBEAR)

**(Optional) Skip the password prompt on future SSH connections:**

```bash
ssh-keygen -t ed25519 -C "your.email@bham.ac.uk"   # default save location is fine
ssh-copy-id YOUR_USERNAME@bluebear.bham.ac.uk       # asks for your password one last time
ssh YOUR_USERNAME@bluebear.bham.ac.uk               # should now connect without a prompt
```

**3. Get the repo (if it is not already cloned)**

This should already be cloned at `/rds/projects/d/dilucam-arme/sam3-background-removal` — you likely just need to `cd` there and `git pull` rather than clone fresh:
```bash
cd /rds/projects/d/dilucam-arme/sam3-background-removal
git pull
```

Note: I generally prefer making updates through a standard code editor like VSCode locally e.g. to `config.json` or any slurm scripts and then pushing changes to GitHub and pulling to the HPC, but you can also edit through the HPC itself using `nano` or `vim` etc in the terminal. 


If the repo doesn't exist yet:
```bash
cd /rds/projects/d/dilucam-arme
git clone git@github.com:arme-project/sam3-background-removal.git
```

**4. Load required modules and set paths**

*IMPORTANT NOTE: (This step is usually handled automatically inside the Slurm scripts — you only need to run it manually if working interactively on the cluster / doing any updates to the virtual environment. If you generate any new Slurm scripts this step will also be important. So this section can mostly be **ignored** if you are just using pre-existing slurm scripts and `sbatch`)*

Run this at the start of every fresh SSH session, before creating or activating the virtual environment. Skipping this causes `pip install` to target the wrong Python environment, leading to confusing "missing package" errors later.
```bash
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HOME=${SAM3_PROJECT_ROOT}/hf_cache
```
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

You're ready to run the pipeline — see [Running the Pipeline](#5-running-the-pipeline) below.



## 5. Running the Pipeline

### Local

Works the same way on both environments once setup is complete. All scripts read from `config.json` in the current directory by default - pass `--config path/to/other.json` to override.

Scripts can be run manually in succession:
```bash
python src/setup_dirs.py
python src/decode_video.py
python src/segment_frame.py
python src/merge_masks.py
python src/remove_fg_noise.py
python src/fill_small_holes.py
python src/opening.py
python src/erosion.py
python src/smooth_gaussian.py
python src/composite_and_stitch.py
python src/cleanup_tmp.py
```

For non-default `config.json`:
```
python src/setup_dirs.py --config path/to/other.json
python src/decode_video.py --config path/to/other.json
python src/segment_frame.py --config path/to/other.json
python src/merge_masks.py --config path/to/other.json
python src/remove_fg_noise.py --config path/to/other.json
python src/fill_small_holes.py --config path/to/other.json
python src/opening.py --config path/to/other.json
python src/erosion.py --config path/to/other.json
python src/smooth_gaussian.py --config path/to/other.json
python src/composite_and_stitch.py --config path/to/other.json
python src/cleanup_tmp.py --config path/to/other.json
```

**But it is recommended to run the full script with:**
`./run_local.sh`
As this is quicker

You can also specify a different configuration file:

`./run_local.sh --config path/to/other.json`

If you have already completed some stages, you can start from a specific stage using --from-stage:

`./run_local.sh --from-stage opening`

Valid stages are:

```
setup
decode
segment
merge
remove_fg_noise
fill_small_holes
opening
erosion
smooth_gaussian
composite
cleanup
```

If run_local.sh is not executable, run it with:

`bash run_local.sh`



### BlueBEAR

On BlueBEAR, steps are typically chained together and submitted via Slurm rather than run interactively.

#### Chained jobs (recommended)

`run_bluebear.sh` submits three dependent Slurm jobs from `slurm/`:

1. `slurm/decode.sh` (CPU): `setup_dirs.py` then `decode_video.py`.
2. `slurm/segment.sh` (A100 GPU): `segment_frame.py` then `merge_masks.py`.
3. `slurm/postprocess.sh` (CPU): `remove_fg_noise.py` through `cleanup_tmp.py`.

Splitting the work this way keeps the scarce GPU allocation for the stage that needs it (segmentation with SAM3). 

Steps:

1. Upload the video to `/rds/projects/d/dilucam-arme/sam3-background-removal/Input_Videos/`.
2. Set `video_name` in `config.json` to match the video (without the `.mov` extension).
3. Check the parameters (see [Recommended parameters](#6-selecting-parameters--editing-configuration-files)) and make any desired changes. It is usually good practice to test a very small video clip before processing a large video clip to see which parameters work well.
4. Go to the repo root: `cd /rds/projects/d/dilucam-arme/sam3-background-removal`
5. Run the launcher directly on the login node:
   ```bash
   bash run_bluebear.sh --config config.json
   ```
6. Note the three job IDs it prints and follow them with `squeue -u $USER`.

Do not run `sbatch run_bluebear.sh`. The launcher only calls `sbatch` three times, which takes about a second and is fine on a login node. The heavy work happens inside the three jobs it submits.

If decode or segment fails, the dependent jobs are cancelled automatically (`--kill-on-invalid-dep=yes`). Fix the problem and run the launcher again. Finished stages are skipped.

To process several videos, make one config per video and run the launcher once per config:
```bash
for cfg in configs/*.json; do bash run_bluebear.sh --config "$cfg"; done
```
Each video then gets its own three-job chain.




#### Single job (short test clips only)

```bash
sbatch run_bluebear_full.sh config.json
```

This runs the whole pipeline in one job with a 1 hour limit. The CPU stages hold an A100 idle, and long videos time out. A 78 second video at 60 fps (4,703 frames, three prompts) did not finish segmentation in 1 hour. Note that this script takes the config as a positional argument, while `run_bluebear.sh` uses `--config`.

#### Email notifications

Add two lines to the `#SBATCH` header block at the top of each script in `slurm/`:
```
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@bham.ac.uk
```
They only work inside the header block, before the first command. Putting them in `run_bluebear.sh` has no effect, because that script is run with `bash` and not submitted with `sbatch`.

Make sure they are updated to match your email - they are currently set to mine.

#### Clean-up (optional)

Safe to delete once the final videos are done:

- `tmp/`: intermediate frames and masks. Deleting it clears the stage cache, so any later rerun starts from decode.
- Stray `slurm-*.out` and `slurm-*.stats` files in the repo root, which come from jobs submitted without an `--output` line. Skim them for errors first.
- Old files in `logs/`. Keep the folder itself.

Keep:

- `sam3-env/` and `hf_cache/`. They are large but required for running. Rebuilding means reinstalling from `requirements.txt` and downloading the model again.



## 6. Selecting Parameters / Editing Configuration Files

Each run reads one config file, `config.json` by default. Pass `--config path/to/other.json` to use a different one. Two configs for different videos differ only in `video_name`.

Key fields:

- `video_name`: the filename of the video in `Input_Videos`, without the extension (for `VA_JH03.mov` use `VA_JH03`). Avoid characters like `#` in filenames, since they break unquoted shell commands.
- `prompts.person`, `prompts.violin`, `prompts.violin_bow`: each has a `threshold` and a `mask_threshold`.
- `postprocess`: settings for `fill_small_holes`, `opening`, `erosion` and `smooth_gaussian`.
- `cleanup.delete_tmp_after_run`: when true, `tmp/` is deleted at the end, which also throws away the stage cache.

Which parameters are best to select / tune?

```jsonc
{
  "video_name": "VN1_RC#02",        // change this to match the video you want to process
  "paths": {                        // these can usually remain the same
    "input_dir": "Input_Videos",
    "output_dir": "Output_Videos",
    "tmp_dir": "tmp"
  },
  "prompts": {                      // prompts can be adjusted depending on the type of video you want to segment
    "person": {
      "short": "p",
      "threshold": 0.3,             // detection confidence cutoff, higher keeps fewer detections
      "mask_threshold": 0.25        // mask pixel cutoff, lower grows the mask, 0.25 is usually decent
    },
    "violin": {
      "short": "v",
      "threshold": 0.3,
      "mask_threshold": 0.3
    },
    "violin_bow": {
      "short": "vb",
      "threshold": 0.3,
      "mask_threshold": 0.3
    }
  },
  "postprocess": {
    "fill_small_holes": {           // fills in small holes in the mask
      "max_hole_area": 30
    },
    "opening": {                    // opens up the mask by removing small specks
      "kernel_size": 2,
      "iterations": 1
    },
    "erosion": {                    // shrinks the mask edge inward, more if you increase the kernel size or iterations (0 disables it)
      "kernel_size": 5,
      "iterations": 1
    },
    "smooth_gaussian": {            // smooths the mask
      "sigma": 2
    }
  },
  "cleanup": {
    "delete_tmp_after_run": false   // keeps temporary intermediate files after completion - useful for debugging
  }
}
```

Sometimes erosion is undesirable - it is better for videos where it is common for the background to show through at the edges of the mask. Iterations can be set to 0 to prevent this.


## 7. Troubleshooting

`squeue -u $USER` can be used to check the queue 

> How to check RDS project quota?

Cloning videos from OneDrive etc.
Videos are now available again on RDS and have been restored - can simply clone them from there, that is probably easier



If you have any questions feel free to reach out: `marci.talbot@bham.ac.uk` 
(I will reply if my email is still active)