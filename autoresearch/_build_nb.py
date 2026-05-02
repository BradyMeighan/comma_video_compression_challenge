"""Build colab_train.ipynb as valid Jupyter JSON."""
import json

CELLS = []

def md(*lines):
    src = "\n".join(lines)
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _to_lines(src)})

def code(*lines):
    src = "\n".join(lines)
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _to_lines(src)})

def _to_lines(s):
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


md(
"# Comma Video Compression — Continue Training (H100 / A100)",
"",
"Resume training from `gen_continued.pt` (score 0.3318 after 173 joint-only epochs on 3090). Uses the SAME `continue_train.py` script and SAME hyperparameters that produced gen_continued.pt earlier today on a 3090 — just bigger batch size and longer wallclock.",
"",
"**Why continue further:** local 3090 run dropped score by ~0.06 in 4h and pose was still falling when the budget ran out. With H100/A100 we get more effective training per wallclock hour.",
"",
"**Plan:**",
"1. Verify A100/H100 is assigned",
"2. Clone repo (includes `autoresearch/colab_run/gen_continued.pt`, ~400KB)",
"3. Set config (longer budget, bs=16 user-validated on A100)",
"4. Run `continue_train.py` (joint-only stage, warm-restart cosine LR, EXACT same script as 3090 run)",
"5. Save new model weights back to Drive",
)

md("## 1. Verify GPU")

code(
"!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv",
"# Need A100 or H100. T4 too slow. Switch via Runtime → Change runtime type.",
)

md("## 2. Mount Google Drive (for output storage)")

code(
"from google.colab import drive",
"drive.mount('/content/drive')",
"",
"import os",
"OUTPUT_DIR = '/content/drive/MyDrive/comma_compression_runs'",
"os.makedirs(OUTPUT_DIR, exist_ok=True)",
"print(f'Outputs will save to: {OUTPUT_DIR}')",
)

md(
"## 3. Clone repo",
"",
"Includes `videos/0.mkv`, `models/*.safetensors`, AND `autoresearch/colab_run/gen_continued.pt` (the resume checkpoint).",
)

code(
"# ─── Set these to your repo ───",
"REPO_URL = 'https://github.com/BradyMeighan/comma_video_compression_challenge.git'",
"BRANCH   = 'autoresearch/apr29'",
"GH_TOKEN = ''  # Personal Access Token if private. Leave empty if public.",
"",
"# Clone",
"!cd /content && rm -rf comma_video_compression_challenge",
"if GH_TOKEN:",
"    url_with_token = REPO_URL.replace('https://', f'https://{GH_TOKEN}@')",
"    !cd /content && git clone --branch {BRANCH} --single-branch {url_with_token}",
"else:",
"    !cd /content && git clone --branch {BRANCH} --single-branch {REPO_URL}",
"",
"%cd /content/comma_video_compression_challenge",
"!ls -la videos/ models/ autoresearch/colab_run/gen_continued.pt",
)

md("## 4. Install dependencies")

code(
"!pip install -q einops safetensors timm segmentation_models_pytorch av brotli",
"!python -c \"import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())\"",
)

md(
"## 5. ⚙️ CONFIG — knobs",
"",
"**These are the EXACT hyperparameters from the 3090 run that produced gen_continued.pt** (score 0.3318 after 173 epochs in 12000s). Only changes vs that run:",
"- `BATCH_SIZE`: 4 → 16 (A100 user-validated at bs=16 with ~65GB VRAM used)",
"- `TRAIN_BUDGET_SEC`: 12000 → longer (we want to push further)",
"",
"LR strategy: warm restart at original `JT_LR=5e-5` (Lion uses lr/3 → 1.67e-5 effective), cosine decay over budget. Same as 3090 run.",
)

code(
"# ─── Resume from this checkpoint (in repo) ───",
"RESUME_FROM = 'autoresearch/colab_run/gen_continued.pt'",
"",
"# ─── Training budget (wallclock seconds) ───",
"# Smoke test:    300   (5 min — verify pipeline)",
"# Match 3090 run: 12000 (3.3h, ~173 epochs at bs=4; should be ~700 epochs at bs=16 on A100)",
"# Production:    36000 (10h — push pose further) ← DEFAULT",
"# Long run:      54000 (15h — diminishing returns but worth trying)",
"TRAIN_BUDGET_SEC = 36000  # 10h production run. Drop to 300 if you want smoke test.",
"",
"# ─── Hyperparameters (EXACT MATCH to 3090 continue_train.py defaults) ───",
"JT_LR_OVERRIDE          = '5e-5'   # SAME as 3090 (Lion takes lr/3 → 1.67e-5 effective)",
"POSE_WEIGHT             = 60.0     # SAME as 3090 (default in continue_train.py)",
"EMA_DECAY               = 0.999    # SAME as 3090",
"COSINE_LR               = 1        # SAME as 3090",
"GRAD_CLIP               = 0.5      # SAME as 3090",
"CHECKPOINT_INTERVAL_SEC = 1800     # save EMA every 30min to .ckpt (overwrites, latest only)",
"CHECKPOINT_EPOCH_INTERVAL = 200    # save EMA every 200 epochs to .e{N}.ckpt (numbered history)",
"CONFIG                  = 'B'      # boundary-weighted focal + Lion (validated winner)",
"FULL_DATA               = 1        # all 600 pairs",
"",
"# ─── Batch size ───",
"# T4 16GB:    bs=2 (smoke only)",
"# 3090 24GB:  bs=4 (what produced gen_continued.pt)",
"# A100 40GB:  bs=8 (safe) or bs=12",
"# A100 80GB:  bs=16 ✓ user-validated, ~65GB VRAM",
"# H100 80GB:  bs=16 (safe, same memory as A100 80GB)",
"BATCH_SIZE = 2  # ⚠️ 2 FOR T4 SMOKE, 16 FOR A100/H100",
"",
"# ─── Output paths ───",
"RUN_NAME    = f'continue_jt{JT_LR_OVERRIDE}_pw{int(POSE_WEIGHT)}_bs{BATCH_SIZE}_b{TRAIN_BUDGET_SEC}'",
"RUN_DIR     = f'{OUTPUT_DIR}/{RUN_NAME}'",
"SAVE_MODEL  = f'{RUN_DIR}/gen.pt'",
"LOG_PATH    = f'{RUN_DIR}/run.log'",
"os.makedirs(RUN_DIR, exist_ok=True)",
"print(f'Resume from: {RESUME_FROM}')",
"print(f'Save to:     {SAVE_MODEL}')",
"print(f'Log:         {LOG_PATH}')",
"print(f'Budget:      {TRAIN_BUDGET_SEC}s ({TRAIN_BUDGET_SEC/3600:.1f}h)')",
"print(f'Hyperparams: JT_LR={JT_LR_OVERRIDE} POSE_WEIGHT={POSE_WEIGHT} EMA={EMA_DECAY} BS={BATCH_SIZE}')",
)

md(
"## 6. (Optional) Smoke test — 5 min to verify resume works",
"",
"Loads checkpoint, runs joint stage for 300s. Confirms data + model wire correctly before committing to a long run.",
)

code(
"import os, subprocess, time",
"env = os.environ.copy()",
"env.update({",
"    'PYTHONUNBUFFERED': '1',",
"    'CONFIG': CONFIG,",
"    'FULL_DATA': str(FULL_DATA),",
"    'MODEL_PATH': RESUME_FROM,",
"    'SAVE_MODEL_PATH': f'{RUN_DIR}/gen_smoke.pt',",
"    'TRAIN_BUDGET_SEC_OVERRIDE': '300',",
"    'JT_LR_OVERRIDE': JT_LR_OVERRIDE,",
"    'POSE_WEIGHT': str(POSE_WEIGHT),",
"    'EMA_DECAY': str(EMA_DECAY),",
"    'COSINE_LR': str(COSINE_LR),",
"    'GRAD_CLIP_OVERRIDE': str(GRAD_CLIP),",
"    'CHECKPOINT_INTERVAL_SEC': str(CHECKPOINT_INTERVAL_SEC),",
"    'CHECKPOINT_EPOCH_INTERVAL': str(CHECKPOINT_EPOCH_INTERVAL),",
"    'BATCH_SIZE': str(BATCH_SIZE),",
"})",
"print(f'Smoke test (300s, bs={BATCH_SIZE})...')",
"print('Watch for: baseline eval, joint stage epoch logs, final eval.')",
"print('=' * 60)",
"t0 = time.time()",
"proc = subprocess.Popen(['python', 'autoresearch/continue_train.py'], env=env,",
"                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
"for line in proc.stdout:",
"    print(line, end='')",
"proc.wait()",
"print('=' * 60)",
"print(f'Done in {time.time()-t0:.1f}s. Exit code: {proc.returncode}')",
"print('Smoke done. Bump TRAIN_BUDGET_SEC + BATCH_SIZE in CONFIG cell, then run cell 7.')",
)

md(
"## 7. Full continue-training run",
"",
"Runs in foreground. Reads `MODEL_PATH` (the resume checkpoint) and writes to Drive:",
"- `{RUN_DIR}/gen.pt` — final EMA-averaged weights",
"- `{RUN_DIR}/gen.pt.ckpt` — latest snapshot (overwrites every 30 min, disconnect insurance)",
"- `{RUN_DIR}/gen.pt.e200.ckpt`, `.e400.ckpt`, ... — numbered snapshots every 200 epochs (history)",
"- `{RUN_DIR}/run.log` — full stdout",
"",
"All ckpts are in your Drive folder so they survive Colab disconnects. If a 10h run dies at hour 8, you have multiple recent checkpoints to resume from.",
)

code(
"import os, subprocess, time",
"env = os.environ.copy()",
"env.update({",
"    'PYTHONUNBUFFERED': '1',",
"    'CONFIG': CONFIG,",
"    'FULL_DATA': str(FULL_DATA),",
"    'MODEL_PATH': RESUME_FROM,",
"    'SAVE_MODEL_PATH': SAVE_MODEL,",
"    'TRAIN_BUDGET_SEC_OVERRIDE': str(TRAIN_BUDGET_SEC),",
"    'JT_LR_OVERRIDE': JT_LR_OVERRIDE,",
"    'POSE_WEIGHT': str(POSE_WEIGHT),",
"    'EMA_DECAY': str(EMA_DECAY),",
"    'COSINE_LR': str(COSINE_LR),",
"    'GRAD_CLIP_OVERRIDE': str(GRAD_CLIP),",
"    'CHECKPOINT_INTERVAL_SEC': str(CHECKPOINT_INTERVAL_SEC),",
"    'CHECKPOINT_EPOCH_INTERVAL': str(CHECKPOINT_EPOCH_INTERVAL),",
"    'BATCH_SIZE': str(BATCH_SIZE),",
"})",
"print(f'Continuing training from {RESUME_FROM}')",
"print(f'Budget: {TRAIN_BUDGET_SEC}s ({TRAIN_BUDGET_SEC/3600:.1f}h) | bs={BATCH_SIZE} | JT_LR={JT_LR_OVERRIDE} | pose_w={POSE_WEIGHT}')",
"print(f'Log: {LOG_PATH}')",
"print(f'Periodic ckpt: {SAVE_MODEL}.ckpt every {CHECKPOINT_INTERVAL_SEC}s')",
"print('=' * 60)",
"t0 = time.time()",
"with open(LOG_PATH, 'w') as f:",
"    proc = subprocess.Popen(['python', 'autoresearch/continue_train.py'], env=env,",
"                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
"    for line in proc.stdout:",
"        print(line, end='')",
"        f.write(line); f.flush()",
"    proc.wait()",
"print('=' * 60)",
"print(f'Done in {time.time()-t0:.1f}s. Exit code: {proc.returncode}')",
)

md(
"## 8. Save model weights to Drive",
"",
"Final EMA-averaged state_dict was already saved to `{SAVE_MODEL}` by `continue_train.py`. Also save code + log for reproducibility.",
)

code(
"import shutil",
"shutil.copy('autoresearch/continue_train.py', f'{RUN_DIR}/continue_train.py')",
"shutil.copy('autoresearch/train.py',          f'{RUN_DIR}/train.py')",
"shutil.copy('autoresearch/prepare.py',        f'{RUN_DIR}/prepare.py')",
"print(f'Saved to {RUN_DIR}:')",
"!ls -la {RUN_DIR}",
"print('Final scores from run.log:')",
"!grep -E '^score:|^seg_term:|^pose_term:|^rate_term:|^model_bytes:|^total_bytes:|^n_params:|^saved_model:' {LOG_PATH}",
)

md(
"## 9. Next steps",
"",
"After this run completes:",
"1. Download `{SAVE_MODEL}` locally, save as `autoresearch/colab_run/gen_continued_v2.pt` (do not overwrite the original).",
"2. Re-run sidecar scripts in `autoresearch/sidecar_*.py` against the new model — lower base pose = more headroom for sidecar gains. Best sidecar config so far: mask K=2 top600 verified greedy + RGB 150_K5+200_K2 (see `autoresearch/sidecar_combined_lean.py`).",
"3. Build submission archive once base + sidecars are finalized.",
)


NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("autoresearch/colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

# Validate
with open("autoresearch/colab_train.ipynb", encoding="utf-8") as f:
    parsed = json.load(f)
print(f"OK: {len(parsed['cells'])} cells, valid JSON")
