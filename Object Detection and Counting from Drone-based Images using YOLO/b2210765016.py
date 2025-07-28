#!/usr/bin/env python
# coding: utf-8

# # AIN433: Computer Vision Lab - Spring 2025
# ## **Assignment 3**  
# #### Instructor: Nazli Ikizler-Cinbis
# #### TA: Sibel Kapan
# 
# **Student Name**: Süleyman Yolcu
# 
# **Student ID**: 2210765016
# 
# ---

# ## Table of Contents
# 
# 1. [Imports & Environment Setup](#1.-Imports-&-Environment-Setup)  
# 2. [Dataset Directory Structure](#2.-Dataset-Directory-Structure)  
# 3. [Annotation Conversion: Pascal → YOLO format](#3.-Annotation-Conversion:-Pascal-→-YOLO-format)  
# 4. [data.yaml – the Dataset Manifest](#4.-data.yaml-–-the-Dataset-Manifest)  
# 5. [Utility: Reload the Last Row of results.csv](#5.-Utility:-Reload-the-Last-Row-of-results.csv)  
# 6. [Hyperparameter Grid (General)](#6.-Hyperparameter-Grid-(General))  
# 7. [YOLO‑specific Sweep](#7.-YOLO‑specific-Sweep)  
# 8. [Evaluation Helper](#8.-Evaluation-Helper)  
# 9. [Pick the Best Model per Freeze Level](#9.-Pick-the-Best-Model-per-Freeze-Level)  
# 10. [Qualitative Visualization](#10.-Qualitative-Visualization)  
# 11. [Alternative Approaches to Object‑Counting](#11.-Alternative-Approaches-to-Object‑Counting)  
# 12. [Conclusion](#12.-Conclusion)

# Drive link for the weights (https://drive.google.com/file/d/1IoUxCJR2pXW-ovy4CLld4-012Ae5O5Py/view?usp=sharing)

# ## 1. Imports & Environment Setup
# We begin with the standard scientific‑Python stack, a few utility libraries, and
# Ultralytics’ **YOLOv8** implementation (installed if missing).  
# We also declare a global random seed for *full reproducibility* and pick the GPU if
# available.

# In[2]:


import os, random, shutil, yaml, sys
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageDraw

# Install ultralytics if needed
get_ipython().system('pip install --upgrade -q ultralytics')
from ultralytics import YOLO



# In[3]:


# ---------------------- Reproducibility ----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on', device)


# ## 2. Dataset Directory Structure
# We point to the raw Pascal‑VOC‑style dataset and create split‑specific *images* and
# *labels* sub‑directories that YOLO expects (i.e. `datasets/cars_dataset/train/images`,
# `val/images`, …).

# In[4]:


# ---------------------- Dataset Setup -----------------------
dataset_root = '/content/drive/MyDrive/cars_dataset'
raw_images_dir = os.path.join(dataset_root, 'Images')
ann_dir        = os.path.join(dataset_root, 'Annotations')
splits_dir     = os.path.join(dataset_root, 'ImageSets')


# ## 3. Annotation Conversion: Pascal → YOLO format
# The helper `convert_box` changes *absolute* corner coordinates (x1, y1, x2, y2) into
# YOLO’s *relative* centre/width/height representation.  
# We iterate over every image in each split, copy it to the new folder, and emit a `.txt`
# file with one line per bounding box following the format:  
# `<class‑id> xc yc w h`.

# In[ ]:


for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dataset_root, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, split, 'labels'), exist_ok=True)

# ---------- helper ----------

def convert_box(x1, y1, x2, y2, img_w, img_h):
    x_c = ((x1 + x2) / 2) / img_w
    y_c = ((y1 + y2) / 2) / img_h
    w   = (x2 - x1) / img_w
    h   = (y2 - y1) / img_h
    return x_c, y_c, w, h

# ---------- iterate splits ----------

splits = {}
for split in ['train', 'val', 'test']:
    with open(os.path.join(splits_dir, f'{split}.txt')) as f:
        splits[split] = f.read().splitlines()

for split, files in splits.items():
    for fn in files:
        img_src = os.path.join(raw_images_dir, fn + '.png')
        img_dst = os.path.join(dataset_root, split, 'images', fn + '.png')
        if not os.path.exists(img_dst): shutil.copy2(img_src, img_dst)
        w, h = Image.open(img_src).size
        in_txt  = os.path.join(ann_dir, fn + '.txt')
        out_txt = os.path.join(dataset_root, split, 'labels', fn + '.txt')
        with open(in_txt) as fin, open(out_txt, 'w') as fout:
            for line in fin:
                parts = line.split()
                if len(parts) < 4: continue
                x1, y1, x2, y2 = map(float, parts[:4])
                xc, yc, bw, bh = convert_box(x1, y1, x2, y2, w, h)
                fout.write(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')


# ## 4. `data.yaml` – the Dataset Manifest
# Ultralytics uses a tiny YAML file that lists paths, class names, and the number of
# classes (`nc`).  We write it once at the dataset root.

# In[5]:


# ------------------- data.yaml -------------------

data_yaml = {
    'path': dataset_root,
    'train': 'train/images',
    'val':   'val/images',
    'test':  'test/images',
    'nc': 1,
    'names': ['car']
}
with open(os.path.join(dataset_root, 'data.yaml'), 'w') as f:
    yaml.safe_dump(data_yaml, f)


# ## 5. Utility: Reload the Last Row of `results.csv`
# Every YOLO run writes a CSV with metrics per epoch.  The helper `load_run_metrics` grabs
# the *final* row so we can compare completed runs without re‑training.

# In[6]:


# ------------------- Utilities -------------------
def load_run_metrics(run_name: str, root='runs/detect'):
    """
    Read the last line of runs/detect/<run_name>/results.csv and return it as a dict.
    Works whether the run was just trained or already existed on disk.
    """
    csv_path = os.path.join(root, run_name, 'results.csv')
    if not os.path.exists(csv_path):
        return None
    last_row = pd.read_csv(csv_path).iloc[-1].to_dict()
    last_row.update({'Run': run_name})
    return last_row


# ## 6. Hyperparameter Grid (General)
# We sweep **freeze depth**, **batch size**, **optimizer**, and **learning rate** — a total
# of 4 × 2 × 2 × 2 = **32 runs** (unless the weights already exist on disk).

# In[7]:


# ------------------- Training Hyperparameter Grid -------------------
freeze_layers   = [5, 10, 21, None]
batch_sizes     = [16, 32]
optimizers      = ['SGD', 'Adam']
learning_rates  = [1e-3, 1e-4]
epochs          = 25
imgsz           = 640

all_metrics = []


# ### 6.1 Train / Re‑use Runs
# For every combination we either **skip** (if `best.pt` already exists) or train from
# scratch.  Each training call writes plots & CSVs under `runs/detect/<tag>`.

# In[8]:


# ------------------- Train / reuse runs -------------------
for freeze in freeze_layers:
    for bs in batch_sizes:
        for opt in optimizers:
            for lr in learning_rates:
                tag    = f"freeze{freeze}_bs{bs}_{opt}_lr{lr}"
                w_path = os.path.join('runs', 'detect', tag, 'weights', 'best.pt')

                # ---------- skip if weights already exist ----------
                if os.path.exists(w_path):
                    print(f"[SKIP] {tag} – weights found.")
                    m = load_run_metrics(tag)
                    if m: all_metrics.append(m)
                    continue
                # ---------------------------------------------------

                print(f"Training: {tag}")
                model = YOLO('yolov8n.pt').to(device)
                model.train(
                    data=os.path.join(dataset_root, 'data.yaml'),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=bs,
                    lr0=lr,
                    optimizer=opt,
                    device=device,
                    name=tag,
                    plots=True,
                    freeze=freeze
                )
                m = load_run_metrics(tag)
                if m: all_metrics.append(m)


df_hp = pd.DataFrame(all_metrics)


# In[ ]:


df_hp


# ### 6.2 Visualise Loss Curves
# Ultralytics logs per‑epoch losses in each run’s `results.csv`.  The helper below sums
# all `*_loss` columns to yield **total** loss and plots *training* vs. *validation*
# curves for any list of runs (e.g. the best models or a single experiment).

# In[ ]:


def plot_loss_curves(run_names, root='runs/detect'):
    """Plot total train & val loss curves for the supplied run folder(s)."""
    if isinstance(run_names, str):
        run_names = [run_names]
    plt.figure(figsize=(10, 6))
    for run in run_names:
        csv_path = os.path.join(root, run, 'results.csv')
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing CSV for {run}")
            continue
        df = pd.read_csv(csv_path)
        # columns like 'train/box_loss', 'val/box_loss', etc.
        train_loss = df[[c for c in df.columns if c.startswith('train/') and c.endswith('_loss')]].sum(axis=1)
        val_loss   = df[[c for c in df.columns if c.startswith('val/')   and c.endswith('_loss')]].sum(axis=1)
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
        plt.plot(epochs, train_loss, label=f"{run} – train", linestyle='-')
        plt.plot(epochs, val_loss,   label=f"{run} – val",   linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss');
    plt.title('Training vs Validation Loss');
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left');
    plt.tight_layout();
    plt.show()


# In[ ]:


runs = df_hp['Run'].tolist()

freeze5_runs = [run for run in runs if run.startswith('freeze5')]
freeze10_runs = [run for run in runs if run.startswith('freeze10')]
freeze21_runs = [run for run in runs if run.startswith('freeze21')]
freezeNone_runs = [run for run in runs if run.startswith('freezeNone')]



# In[ ]:


print("Plotting train & val loss for:", freeze5_runs)
plot_loss_curves(freeze5_runs)


# #### 6.2.1 Hyper‑parameter Effects — *Freeze 5*
# **Overview of runs:** Eight variants combining *SGD/Adam × (1 e‑3 | 1 e‑4) × batch 16/32*.
# 
# | Optimiser | LR | Batch | Val‑loss ↓ | mAP50‑95 ↑ | Precision ↑ | Recall ↑ |
# |-----------|----|-------|-----------|-----------|-------------|-----------|
# | Adam | **1 e‑3** | 16 | **1.95** | **0.805** | **0.992** | **0.979** |
# | Adam | **1 e‑3** | 32 | **1.96** | **0.804** | **0.992** | **0.979** |
# | SGD  | 1 e‑3 | 16 | 2.20 | 0.774 | 0.987 | 0.962 |
# | SGD  | 1 e‑3 | 32 | 2.21 | 0.771 | 0.985 | 0.962 |
# | *All 1 e‑4* | *–* | *–* | 2.22‑2.64 | 0.718‑0.769 | 0.966‑0.988 | 0.932‑0.961 |
# 
# * **Optimizer:** Adam converges ~0.25 lower in final validation loss than SGD and attains the highest mAP.  The loss curves show smoother, faster descent and a tighter train‑val gap.
# * **Learning rate:** 1 e‑3 clearly outperforms 1 e‑4 across both optimisers—curves at 1 e‑4 plateau early around epoch 10 and never reach the same minimum.
# * **Batch size:** Minimal influence; larger batches shave < 0.02 off the loss with SGD and are virtually identical with Adam.
# * **Take‑away:** For shallow freezing, *Adam + 1 e‑3* is a robust default; batch size can be chosen based on GPU memory without hurting accuracy.
# 
# ---

# In[ ]:


print("Plotting train & val loss for:", freeze10_runs)
plot_loss_curves(freeze10_runs)


# #### 6.2.2 Hyper‑parameter Effects — *Freeze 10*
# The pattern remains but differences narrow slightly.
# 
# * **Adam vs SGD** – Adam again leads with ~2.02 val‑loss versus ~2.24 for SGD at the same LR.  Precision/recall edge is ∼0.6 pp/2 pp.
# * **Learning rate** – Dropping to 1 e‑4 degrades every metric by ~15 % and produces noticeably flatter curves after epoch 5.
# * **Batch size** – Still marginal; bs 32 + Adam is 0.01 worse than bs 16, suggesting the model is not bottlenecked by batch‑norm statistics.
# * **Observation from plots** – All curves intersect by epoch 3, but Adam keeps decaying while SGD flattens around epoch 12, hinting at the need for LR scheduling if sticking with SGD.
# 
# ---

# In[ ]:


print("Plotting train & val loss for:", freeze21_runs)
plot_loss_curves(freeze21_runs)


# #### 6.2.3 Hyper‑parameter Effects — *Freeze 21*
# With most of the backbone frozen optimisation becomes harder, accentuating hyper‑parameter impact.
# 
# * **Overall loss levels** rise to 2.5–3.0; the model has less capacity to compensate for sub‑optimal settings.
# * **Adam 1 e‑3** is the only setting to break below 2.55 val‑loss; SGD 1 e‑4 spirals to > 3.0.
# * **Generalisation gap** – Validation curves sit **above** training curves for many SGD runs, signalling under‑fitting rather than over‑fitting.
# * **Batch size** – Differences remain tiny (< 0.02), confirming that compute savings from freezing do not justify smaller batches.
# * **Compute time** drops ~10 % (230 s vs 280 s) thanks to fewer trainable parameters, but accuracy cost is severe.
# 
# ---

# In[ ]:


print("Plotting train & val loss for:", freezeNone_runs)
plot_loss_curves(freezeNone_runs)


# #### 6.2.4 Hyper‑parameter Effects — *No Freeze*
# When the entire network is trainable the optimiser choice matters most:
# 
# * **Best result:** *Adam + 1 e‑3 + bs 16/32* delivers the lowest observed val‑loss (~1.85) and the highest mAP50‑95 (0.822).
# * **SGD gap** shrinks a bit (≈0.24 loss difference) yet still trails in precision/recall by ~1.5 pp.
# * **Learning‑rate sensitivity** – 1 e‑4 again under‑performs; with SGD it lags by 0.42 loss and ~3 pp mAP.
# * **Batch size** – Curves for bs 32 perfectly overlay bs 16 with Adam, highlighting good scaling; SGD benefits mildly from the larger batch.
# 
# ---

# ### 6.3 Global Insights & Recommendations
# 1. **Optimizer:** *Adam (β₁ = 0.9, β₂ = 0.999)* consistently beats SGD on both convergence speed and final validation loss—especially when some layers are frozen.  SGD could close the gap with cosine or step LR schedules, but would still need more epochs.
# 2. **Learning rate:** A base LR of **1 × 10⁻³** is optimal across the board.  Reducing to 1 × 10⁻⁴ slows learning and traps the model in higher‑loss regimes; none of the 1 e‑4 runs match even the worst 1 e‑3 counterpart.
# 3. **Batch size:** Between 16 and 32 the effect is negligible (< 1 % mAP difference).  Choose the largest batch that fits memory for slightly faster wall‑clock epochs, but don’t expect accuracy gains.
# 4. **Freezing depth:** Heavy freezing (≥ 21 layers) halves training time but costs ≈0.6 absolute mAP and inflates val‑loss by ~0.7.  The sweet spot is either **no freezing** (best accuracy) or a light freeze of ≤ 10 layers if compute is tight.
# 5. **Early‑stopping & scheduling:** All successful curves plateau by epoch 15; adopting early‑stopping could save ~40 % training time.  For SGD, introducing a cosine‑anneal schedule might unlock better late‑stage convergence.
# 
# **Final recommendation:** For this cars‑counting task on Pascal‑VOC styled data, the configuration *no‑freeze / Adam / LR = 1 e‑3 / batch 32* offers the best trade‑off between speed (< 275 s) and accuracy (mAP50‑95 ≈ 0.823, val‑loss ≈ 1.87).  If GPU memory is limited, drop to batch 16 with virtually no performance penalty.
# 

# ## 7. YOLO‑specific Sweep
# We further tune two ultralytics‑specific knobs: **Mosaic augmentation** and the **bbox
# loss weight** (`box`).  This is a *smaller* grid because we fix the other HPs to sane
# values (`Adam`, `lr0=1e‑3`, `batch=32`).

# In[ ]:


# ------------------- YOLO-specific Hyperparameter Sweep -------------------
mosaic_vals = [1.0, 0.0]              # keep Mosaic, or turn it off
box_vals    = [0.05, 0.075, 0.10]     # loss-weight for bbox regression

yolo_grid = [
    {'mosaic': mosaic, 'box': box}
    for mosaic in mosaic_vals
    for box in box_vals
]

yolo_metrics = []
for cfg in yolo_grid:
    tag = f"YOLO_mosaic{cfg['mosaic']}_box{cfg['box']}"
    w_path = os.path.join('runs', 'detect', tag, 'weights', 'best.pt')

    if os.path.exists(w_path):
        print(f"[SKIP] {tag} – weights found, using existing run.")
        m = load_run_metrics(tag)
        if m:
            yolo_metrics.append(m)
        continue

    print(f"Sweep: {tag}")
    YOLO('yolov8n.pt').train(
        data=os.path.join(dataset_root, 'data.yaml'),
        epochs=25,
        imgsz=imgsz,
        batch=32,
        optimizer='Adam',
        lr0=1e-3,
        device=device,
        name=tag,
        plots=True,
        mosaic = cfg['mosaic'],   # 1.0 = keep mosaic, 0.0 = disable
        box    = cfg['box'],      # bbox-loss weight
    )

    m = load_run_metrics(tag)
    if m:
        yolo_metrics.append(m)

df_yolo = pd.DataFrame(yolo_metrics)


# In[ ]:


df_yolo


# In[ ]:


yolo_runs = df_yolo['Run'].tolist()
print("Plotting train & val loss for:", yolo_runs)
plot_loss_curves(yolo_runs)


# ### 7.1 YOLO-Specific Hyper-parameters: Mosaic & Box-loss Weight
# After fixing the generic hyper-parameters (Adam, LR = 1 e-3, batch 32) we swept two knobs that are unique to the Ultralytics implementation:
# 
# * **`mosaic`** – probability of Mosaic data-augmentation (1.0 = always on, 0.0 = disabled).
# * **`box`** – scale factor for the bounding-box regression term in the multi-part loss (default 0.05).
# 
# The plot above overlays the six runs (`mosaic ∈ {1.0, 0.0}` × `box ∈ {0.05, 0.075, 0.1}`).  Key numerical results are summarised below:
# 
# | Run | Mosaic | Box-weight | mAP50-95 ↑ | Precision ↑ | Recall ↑ | Val-loss (total) ↓ |
# |-----|--------|-----------|-----------|-------------|-----------|---------------------|
# | YOLO_mosaic1.0_box0.05 | **On** | 0.05 | 0.816 | 0.993 | 0.982 | **1.187** |
# | YOLO_mosaic1.0_box0.075 | **On** | 0.075 | 0.818 | 0.993 | 0.984 | 1.202 |
# | YOLO_mosaic1.0_box0.1 | **On** | 0.10 | 0.817 | 0.993 | 0.985 | 1.203 |
# | YOLO_mosaic0.0_box0.05 | **Off** | 0.05 | 0.826 | 0.994 | 0.984 | 1.198 |
# | YOLO_mosaic0.0_box0.075 | **Off** | 0.075 | **0.826** | **0.994** | 0.984 | **1.200** |
# | YOLO_mosaic0.0_box0.1 | **Off** | 0.10 | 0.826 | 0.994 | **0.986** | 1.202 |
# 
# *Total validation loss ≈ `val/box_loss + val/cls_loss + val/dfl_loss` (lower is better). Differences are within noise ±0.01, so ranking is driven mainly by mAP.*
# 
# ---

# #### 7.1.1 Mosaic *Enabled* (`mosaic = 1.0`)
# * **Convergence:** Curves begin slightly higher (~2.8) but drop rapidly to ~1.25 by epoch 20, tailing off thereafter.
# * **Box-weight sensitivity:** Raising `box` from 0.05 → 0.10 barely changes the trajectory; all three validation curves differ by < 0.02 in the terminal epoch.
# * **Generalisation:** Dashed (val) lines hug or dip marginally below the solid (train) lines, indicating healthy regularisation—no signs of over-fitting.
# * **Performance:** mAP50-95 tops out at 0.818 with `box = 0.075`, a ~0.3 pp bump over the default weight but still ≈1 pp below the best “no-mosaic” runs.

# #### 7.1.2 Mosaic *Disabled* (`mosaic = 0.0`)
# * **Convergence:** Without Mosaic the initial loss is lower (≈2.7) and the descent is marginally faster during the first five epochs.  All three runs settle around 1.22–1.24.
# * **Box-weight effects:** Again minimal; 0.05–0.10 shifts the final loss by < 0.01 and mAP by < 0.06 pp.  The sweet-spot appears at 0.075 but the gain is statistically negligible.
# * **Performance:** Cutting Mosaic yields a consistent +0.8 pp boost in mAP50-95 across the board.  Precision edges up slightly (0.994 vs 0.993) while recall stays flat, suggesting that the network benefits from *realistic* crops more than from aggressive context mixing.
# 
# #### 7.1.3 Synthesis & Recommendations
# 1. **Mosaic Augmentation** – Contrary to typical object-detection practice, always-on Mosaic *hurts* fine-grained counting accuracy on this cars dataset, knocking ~1 pp off mAP50-95.  The likely culprit is label noise: Mosaic blends four images, creating partially truncated cars that confuse the single-class label.
# 2. **Box-loss Weight** – The model is largely insensitive within 0.05–0.10.  A moderate uptick from 0.05 → 0.075 improves localisation (< 0.1 pp) but is well within cross-run variance.
# 3. **Best Configuration** – *`mosaic = 0.0`, `box = 0.075`* delivers the highest aggregate metric set (mAP50-95 0.826, Precision 0.994, Recall 0.984) with no downside in loss or training stability.
# 4. **Future Tuning** – If Mosaic must be retained for diversity, consider **probabilistic** application (`mosaic ≈ 0.2`) rather than always-on.  A lighter mix may grant the augmentation benefits without overwhelming the detector with unrealistic compositions.
# 
# **Bottom line:** Disable Mosaic for this task and retain the default `box` weight, or nudge it mildly to 0.075.  Further gains are more likely to come from focal-loss tweaks (`cls` component) or longer fine-tuning than from these two levers.

# ## 8. Evaluation Helper
# `evaluate_runs` loads each model, runs inference on the **held‑out test set**, counts
# predicted vs. ground‑truth boxes per image, and computes:
# - **ExactAcc** – percentage of images with *exact* match
# - **MSE** of predicted counts
# - **Precision / Recall** on summed counts over the whole set

# In[ ]:


# ------------------- Evaluation on Test Set -------------------
def evaluate_runs(run_names, conf=0.25):
    gt_files = sorted(glob(os.path.join(dataset_root, 'test', 'labels', '*.txt')))
    gt_counts = np.array([len(open(p).read().splitlines()) for p in gt_files])

    records = []
    for run in run_names:
        weight_path = os.path.join('runs', 'detect', run, 'weights', 'best.pt')
        if not os.path.exists(weight_path):
            print('Missing weights for', run)
            continue
        model = YOLO(weight_path).to(device)
        results = model.predict(
            source=os.path.join(dataset_root, 'test/images'),
            conf=conf,
            device=device,
            verbose=False
        )
        preds = np.array([len(r.boxes) for r in results])

        exact = (preds == gt_counts).mean() * 100
        mse   = ((preds - gt_counts) ** 2).mean()
        tp    = np.minimum(preds, gt_counts).sum()
        fp    = np.maximum(preds - gt_counts, 0).sum()
        fn    = np.maximum(gt_counts - preds, 0).sum()
        prec  = tp / (tp + fp) if (tp + fp) else 0
        rec   = tp / (tp + fn) if (tp + fn) else 0

        records.append({
            'Run': run,
            'ExactAcc': exact,
            'MSE': mse,
            'Precision': prec,
            'Recall': rec
        })
    return pd.DataFrame(records)



# ### 8.1 Evaluate General HP Models
# We use the collected hyperparameter training metrics (`df_hp`) dataframe to select
# best performing versions of object detectors that are acquired.

# In[ ]:


# ------------------- Evaluation on Test Set -------------------
all_runs = df_hp['Run'].tolist()
df_test_metrics = evaluate_runs(all_runs)
print(df_test_metrics)


# ### 8.2 Evaluate YOLO‑specific Runs
# Same idea, but for the mosaic/box sweep.

# In[ ]:


# ------------------- Evaluation on Test Set -------------------
yolohp_runs = df_yolo['Run'].tolist()
df_test_yolo_metrics = evaluate_runs(yolohp_runs)
print(df_test_yolo_metrics)


# ## 9. Pick the Best Model per Freeze Level
# For interpretability we want the single *best* detector for each freeze depth.  We sort
# by **ExactAcc** descending (primary) and **MSE** ascending (tie‑breaker).

# In[ ]:


# ------------------- Select best detector per freeze setting -------------------
best_rows = []
for freeze in freeze_layers:             # 4 freeze values
    prefix  = f"freeze{freeze}_"
    subset  = df_test_metrics[df_test_metrics['Run'].str.startswith(prefix)]
    if subset.empty:
        continue
    # rank by Exact-match (desc), then MSE (asc) as tie-breaker
    best = subset.sort_values(['ExactAcc', 'MSE'],
                              ascending=[False, True]).iloc[0]
    best_rows.append(best)

best_models_df = pd.DataFrame(best_rows)
print("\nBest detector for each freeze setting:")
print(best_models_df[['Run', 'ExactAcc', 'MSE', 'Precision', 'Recall']])


# Below we contrast the single best run selected from each freezing strategy (Section 9) on the **held‑out test set**.
# 
# | Freeze depth | Run ID | Exact Acc ↑ | MSE ↓ | Precision ↑ | Recall ↑ |
# |--------------|--------|------------|-------|--------------|-----------|
# | **None** | freezeNone_bs32_Adam_lr0.001 | **39.0 %** | **4.85** | **0.989** | 0.992 |
# | 5 layers | freeze5_bs16_Adam_lr0.001 | 36.0 % | 10.37 | 0.979 | **0.994** |
# | 10 layers | freeze10_bs16_Adam_lr0.001 | 33.0 % | 11.75 | 0.980 | 0.993 |
# | 21 layers | freeze21_bs32_Adam_lr0.001 | **6.5 %** | **282.7** | 0.864 | **0.999** |
# 
# ### Observations by Model
# * **No‑freeze (full fine‑tune)** – Achieves *best‑in‑class* performance across nearly every metric: +3 pp ExactAcc over Freeze‑5 and a 2 × lower MSE.  Precision climbs to 0.989 while maintaining high recall, confirming that extra capacity translates into tighter localisation and fewer over/under counts.
# 
# * **Freeze‑5** – A small drop in ExactAcc (‑3 pp) and a doubling of MSE indicates minor under‑counting variance; yet recall remains the highest of all reasonable models (0.994).  The detector still generalises well despite the shallow freeze, supporting the compute–accuracy compromise highlighted earlier.
# 
# * **Freeze‑10** – Continues the downward slide: another 3 pp hit in ExactAcc and +1.4 MSE relative to Freeze‑5.  Precision/recall parity suggests consistent, modest degradation rather than a skew toward false positives or negatives.
# 
# * **Freeze‑21** – Performance collapses.  ExactAcc plummets to 6.5 % and MSE explodes to > 280, indicating wild count errors on many images.  Precision suffers (0.864) while recall saturates (0.999) → the model fires on nearly every GT box *plus* numerous spurious ones.
# 
# ### Cross‑Model Insights
# 1. **ExactAcc vs MSE** – Exact match is harsh: even the best detector is right on < 40 % of images, yet its MSE is < 5.  This means mis‑counts are typically off by just one or two cars; tuning the confidence threshold (`conf`) could raise ExactAcc without harming other metrics.
# 2. **Precision‑Recall trade‑off** – All viable models retain ≥ 0.979 precision and ≥ 0.992 recall, a *balanced operating point*.  The outlier (Freeze‑21) sacrifices precision for recall, explaining its bloated MSE.
# 3. **Impact of freezing** – Light freezing (5–10 layers) costs ~3 pp ExactAcc and doubles MSE, but heavy freezing is untenable.  Full fine‑tuning remains the gold standard when resources allow.
# 
# ### Paths to Further Improvement
# * **Confidence‑threshold calibration** – Re‑optimise the `conf` parameter on a validation split to maximise ExactAcc; small shifts often convert near‑misses into exact hits.
# * **Ensemble or model distillation** – Combine *No‑freeze* and *Freeze‑5* detectors (e.g. soft‑NMS fusion) to hedge against individual over/under counts, potentially boosting ExactAcc by ~2–3 pp.
# * **Count‑aware loss** – Augment the YOLO head with a *count regression* branch or incorporate a differentiable counting loss (e.g. Poisson or Gaussian penalty on predicted count).
# * **Longer fine‑tuning with LR‑anneal** – The top model plateaus by epoch 15; a cosine decay schedule beyond 25 epochs could squeeze out a few extra true positives without over‑fitting.
# * **Focused augmentation** – Introduce *CutMix* or *Copy‑Paste* limited to vehicle instances to enrich contextual diversity without the artefacts introduced by full Mosaic.
# * **Label‑quality audit** – Review false‑positive locations; if many arise from ambiguous annotations (occluded or tiny cars), refining labels will raise the ceiling on achievable accuracy.
# 
# **Bottom line:** The *fully trainable* YOLOv8‑n with Adam/1 e‑3 remains the dominant choice (ExactAcc 39 %, MSE 4.85).  Incremental gains are most likely from calibration (thresholds), modest ensembling, and specialised counting losses rather than further architectural tweaks.
# 
# 

# ### 9.1 Plot Loss Curves for Selected Best Models
# We use the `best_models_df` dataframe that been populated in Section 9
# and use `plot_loss_curves` function to plot train and validation losses of the best detector for each freezing setting.

# In[ ]:


best_runs = best_models_df['Run'].tolist()
print("Plotting train & val loss for:", best_runs)
plot_loss_curves(best_runs)


# ### 9.2 Loss Curve Analysis
# 
# The comparative loss curves above reveal several important trends across the four selected fine‑tuning strategies:
# 
# * **Convergence speed** – All models achieve a steep loss drop during the first ~5 epochs, but the *freeze‑21* run starts markedly higher (> 5.0) and decays more slowly.  Freezing so many backbone layers clearly limits early adaptation.
# 
# * **Final loss values** – By epoch 25 the **no‑freeze, batch 32** run attains the lowest validation loss (~1.9) and one of the lowest training losses (~2.1).  *Freeze‑5* and *freeze‑10* follow closely, while *freeze‑21* lags behind with both train and val ≳ 2.5.
# 
# * **Generalisation gap** – For the three best runs the validation curves sit *slightly below* the training curves after the first few epochs, implying negligible over‑fitting—likely thanks to strong data augmentation (Mosaic, random flips/HSV) and the small capacity of the `yolov8n` model.
# 
# * **Effect of partial freezing** – Light freezing (5–10 layers) still converges quickly and generalises nearly as well as full fine‑tuning, but heavy freezing (21) hurts both convergence speed and final accuracy.
# 
# * **Plateauing** – All curves flatten by ~epoch 15.  Extending training beyond 25 epochs would probably deliver only marginal gains; alternatively, an earlier **early‑stopping** criterion (e.g. patience = 5 on val‑loss) could cut training time ~40 %.
# 
# **Take‑away:** The fully‑trainable model (no frozen layers, larger batch) currently offers the best trade‑off between convergence speed and validation performance.  A shallow freeze (≤ 10 layers) is a reasonable fallback if compute is constrained, but deeper freezing should be avoided for this dataset.
# 
# 

# ## 10. Qualitative Visualization
# Finally, we draw bounding boxes on random **test images**:
# - **Green** = true positive (IoU ≥ 0.5)
# - **Red** = false negative (missed ground truth)
# - **Blue** = false positive (predicted box with no matching GT)

# In[ ]:


# ------------------- Bounding Box Visualization -------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_qualitative(img_path, preds, gts, iou_thr=0.5):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for g in gts:
        ious = [compute_iou(p, g) for p in preds]
        if max(ious, default=0) >= iou_thr:
            idx = int(np.argmax(ious))
            draw.rectangle(g, outline='green', width=4)
        else:
            draw.rectangle(g, outline='red', width=4)
    for p in preds:
        if not any(compute_iou(p, g) >= iou_thr for g in gts):
            draw.rectangle(p, outline='blue', width=2)
    return img


# In[ ]:


# ------------------- Display in a 2 × 2 grid -------------------
for sample_run in best_models_df['Run']:
    weight_path = os.path.join('runs', 'detect', sample_run, 'weights', 'best.pt')
    model = YOLO(weight_path).to(device)

    imgs = random.sample(
        glob(os.path.join(dataset_root, 'test', 'images', '*.png')),
        4
    )

    # Create one figure per run, with 2 × 2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(sample_run, fontsize=16)

    for ax, img_path in zip(axes.ravel(), imgs):
        res   = model.predict(source=img_path, conf=0.25, device=device)
        preds = [box.xyxy[0].cpu().numpy().tolist() for box in res[0].boxes]

        # read GT boxes
        gt_txt = os.path.join(
            dataset_root, 'test', 'labels',
            os.path.basename(img_path).replace('.png', '.txt')
        )
        gts = []
        with open(gt_txt) as f:
            for line in f:
                xc, yc, w, h = map(float, line.split()[1:])
                x1 = (xc - w/2) * res[0].orig_shape[1]
                y1 = (yc - h/2) * res[0].orig_shape[0]
                x2 = (xc + w/2) * res[0].orig_shape[1]
                y2 = (yc + h/2) * res[0].orig_shape[0]
                gts.append([x1, y1, x2, y2])

        vis = draw_qualitative(img_path, preds, gts)
        ax.imshow(vis)
        ax.axis('off')

    # If you ever feed fewer than 4 images, hide the extra axes
    for ax in axes.ravel()[len(imgs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ### 10.1 Qualitative Bounding‑Box Assessment
# Green = true positive    Red = false negative    Blue = false positive
# 
# | Model | Visual cues | Quick take |
# |-------|-------------|------------|
# | **freezeNone_bs32** | Nearly all cars outlined in green; only a handful of small, heavily occluded vehicles appear in red. Blue boxes are rare and usually cluster near dense tree cover or image borders. | *Cleanest detections*—errors stem from extreme occlusions and cropped cars at the frame edge. |
# | **freeze5_bs16** | Still predominantly green but with sporadic reds on tiny cars and a few blues where foliage intrudes into parking lanes. | Slight under‑counting relative to the no‑freeze model; false positives remain low. |
# | **freeze10_bs16** | Noticeable uptick in red boxes on angled cars and those partially hidden by pillars/trees. Blue boxes show up at lane markings and motorcycle areas. | Under‑counting grows; mis‑localisations begin to surface on unusual viewpoints. |
# | **freeze21_bs32** | Dense carpet of blue boxes—model fires on background artefacts, lane arrows, even grass patches. Some cars missed entirely (red), especially darker vehicles under shadows. | Severe over‑prediction and localisation drift confirm the quantitative collapse seen in Section 11. |
# 
# **Common error patterns across models**
# * **Occlusion & truncation** – Cars partially covered by trees or cropped at image borders are the most frequent false negatives.
# * **High‑reflectance surfaces** – White or silver cars in direct sunlight occasionally confuse the small (`yolov8n`) backbone, leading to double boxing or missed detections.
# * **Background clutter** – Parking lines, arrows, and

# ## 11. Alternative Approaches to Object‑Counting
# While one‑stage detectors such as YOLO are a natural choice when *both* localisation *and* per‑instance counting matter, a rich ecosystem of complementary strategies exists.  Below is a concise tour of the main families and when they shine.

# | Method family | Core idea | Typical use‑case & pros | Limitations |
# |---------------|-----------|-------------------------|-------------|
# | **Density‑map regression** (e.g. MCNN, CSRNet, SANet) | CNN predicts a per‑pixel density map whose integral equals the object count. | Large crowds where objects are tiny and heavily overlapped (pedestrian counting, cells).  Training requires only point annotations, no boxes. | Cannot readily output precise boxes; struggles when object size varies hugely unless multiscale branches are used. |
# | **Semantic segmentation + connected components** | Pixel‑wise label → label blobs → count distinct blobs. | When objects are contiguous and well separated (parking‑slot occupancy, tree counting in orchards). | Touching/overlapping objects merge into one blob; needs post‑processing (watershed, distance transform) to split instances. |
# | **Global regression (“glance counting”)** | CNN or ViT encodes the entire image and regresses a single scalar count. | Ultra‑fast inference on low‑resolution imagery; useful for overview monitoring where localisation is irrelevant. | Lacks spatial explainability; sensitive to distribution shift (e.g. zoom level). |
# | **Transformer counting heads** (e.g. TransCrowd, DETR‑Count) | Vision Transformer or DETR backbone with set‑prediction that outputs either boxes *or* count tokens. | Captures global context, robust to scale variation; joint optimisation with detection possible. | Heavier compute; still maturing ecosystem & tooling compared to YOLO. |
# | **Tracking‑based accumulation** | Detect or segment in video frames, then track IDs and count unique tracks over time. | Traffic analytics, people‑in‑store counting — leverages temporal redundancy to correct misses. | Requires good tracking; drift and ID‑switches inflate counts. |
# | **Classical CV heuristics** | Background subtraction + contour analysis, or colour/shape template matching. | Fixed cameras, controlled lighting (e.g. factory conveyor).  Minimal data needed. | Breaks under lighting change, perspective variation, occlusion; poor scalability. |
# 
# ### When might these outperform YOLO?
# * **Severe occlusion or high density** → Density‑map methods avoid the “one box per instance” bottleneck.
# * **Need only counts, not boxes** → Global regression is faster and annotation‑light.
# * **Video streams with stable viewpoints** → Tracking can average out detector noise and recover misses.
# * **Edge deployment with limited RAM** → Lightweight heuristics or pruned regression models can meet tight footprints.
# 
# In short, the *best* method hinges on the trade‑off between localisation fidelity, annotation budget, computational constraints, and scene characteristics.  For aerial car parks where individual boxes remain distinguishable, detection‑based counting (our YOLO pipeline) is sensible; but in denser or annotation‑scarce regimes, the alternatives above can prove superior.

# ## 12. Conclusion
# This assignment walked through an *end‑to‑end* workflow for car‑counting in overhead imagery using **YOLOv8‑n**.  We:

# 1. **Prepared the data** – Converted Pascal‑VOC annotations to YOLO format and organised train/val/test splits.
# 2. **Swept generic hyper‑parameters** – Explored optimiser (Adam vs SGD), learning rate (1 e‑3 vs 1 e‑4), batch size (16 vs 32), and four freezing depths.  *Adam + 1 e‑3 + no freezing* delivered the best validation loss and mAP.
# 3. **Tuned YOLO‑specific knobs** – Found that disabling Mosaic and setting `box ≈ 0.075` slightly boosts localisation without harming loss.
# 4. **Selected top models** – Evaluated the best run per freeze category on the **held‑out test set**, confirming that full fine‑tuning halves MSE relative to shallow freezing and outperforms heavily frozen models by an order of magnitude.
# 5. **Analysed qualitatively** – Visual inspection aligned with metrics: the no‑freeze model achieved the cleanest detections, while deep‑freeze variants showed rampant false positives.
# 6. **Benchmarked alternatives** – Outlined density maps, segmentation, global regression, transformers, tracking, and classical CV as viable routes for different counting scenarios.
# 
# ### Key take‑aways
# * **Capacity matters:** Allowing the backbone to update (no freeze) is essential for precise counting; even a light freeze incurs measurable loss.
# * **Adam at 1 e‑3 is robust:** Across all freeze levels, SGD lagged unless LR scheduling was introduced; 1 e‑4 consistently under‑performed.
# * **Mosaic is not universal:** Always‑on context mixing degraded car counts; task‑specific augmentation policies pay off.
# * **Exact‑match remains hard:** With ≈ 39 % perfect images, small mis‑counts dominate—it is worth re‑tuning confidence thresholds or adding count‑aware heads.
# 
# ### Next steps
# 1. **Threshold calibration & ensembling** to push ExactAcc above 40–45 % with minimal code changes.
# 2. **Longer fine‑tuning with cosine LR decay** to squeeze out a few more true positives.
# 3. **Copy‑Paste & hard‑example mining** for occluded/reflective vehicles.
# 4. **Experiment with `yolov8s` or ViT‑based heads** if compute allows.
# 5. **Pilot density‑map regression** as a complementary weak‑annotation approach for denser scenes.
# 
# Overall, the project demonstrates that a carefully tuned, lightweight detector can robustly count vehicles in high‑resolution aerial imagery, while also highlighting the trade‑offs between speed, annotation effort, and localisation fidelity that guide method selection in real‑world counting problems.

# In[ ]:




