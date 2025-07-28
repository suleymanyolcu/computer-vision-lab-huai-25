#!/usr/bin/env python
# coding: utf-8

# # AIN433: Computer Vision Lab - Spring 2025
# ## **Assignment 2**  
# #### Instructor: Nazli Ikizler-Cinbis
# #### TA: Sibel Kapan
# 
# **Student Name**: Süleyman Yolcu
# 
# **Student ID**: 2210765016
# 

# # PART 1
# ---

# # 0. Setup & Imports

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


from pathlib import Path
import time, copy, math, itertools, gc, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED);  torch.backends.cudnn.deterministic = True
print('Running on:', DEVICE)


# # 1. Data Loading & Augmentation

# In[3]:


ROOT = Path('/content/drive/MyDrive/Colab Notebooks/food11')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# In[4]:


def get_loaders(batch_size: int = 32, num_workers: int = 4, pin=True):
    train_ds = datasets.ImageFolder(ROOT/'train',      transform=train_tf)
    val_ds   = datasets.ImageFolder(ROOT/'validation', transform=eval_tf)
    test_ds  = datasets.ImageFolder(ROOT/'test',       transform=eval_tf)

    return (
        {
            'train': DataLoader(train_ds, batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin),
            'val'  : DataLoader(val_ds,   batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin),
            'test' : DataLoader(test_ds,  batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin),
        },
        train_ds.classes
    )


# # 2. Model Definitions  
# Here we declare two *distinct* CNN architectures that share low-level
# building blocks but diverge in their topologies.
# 
# * `PlainCNN5` – a straightforward 5-conv network  
# * `ResCNN5`   – identical stem, but last two stages become residual blocks
# 

# In[5]:


# ---- shared building blocks --------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, pool=False):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if pool: layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)

class ResidualBlock(nn.Module):
    """2×3×3 conv with identity/1×1 projection skip."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c)
        self.conv2 = ConvBNReLU(out_c, out_c)
        self.proj  = nn.Identity() if in_c==out_c else nn.Conv2d(in_c, out_c, 1, bias=False)
    def forward(self, x): return self.conv2(self.conv1(x)) + self.proj(x)

# ---- model A: plain -----------------------------------------------------------
class PlainCNN5(nn.Module):
    """5 conv blocks → 2 FC layers."""
    def __init__(self, n_classes=11, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3,   32, pool=True),   # 224→112
            ConvBNReLU(32,  64, pool=True),   # 112→56
            ConvBNReLU(64, 128, pool=True),   # 56 →28
            ConvBNReLU(128,256, pool=True),   # 28 →14
            ConvBNReLU(256,512, pool=True),   # 14 →7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ---- model B: residual --------------------------------------------------------
class ResCNN5(nn.Module):
    """3 conv blocks + 2 residual blocks → 2 FC layers."""
    def __init__(self, n_classes=11, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3,   32, pool=True),   # 224→112
            ConvBNReLU(32,  64, pool=True),   # 112→56
            ConvBNReLU(64, 128, pool=True),   # 56 →28
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),                  # 28→14
            ResidualBlock(256, 512),
            nn.MaxPool2d(2),                  # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


# In[6]:


def count_params(model):
    """Return #trainable parameters (millions)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# # 3. Training & Evaluation Utilities
# 

# In[7]:


def epoch_loop(model, loader, criterion, optimiser=None):
    train = optimiser is not None
    model.train(train)

    total_loss = correct = total = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        if train: optimiser.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                optimiser.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    return total_loss/total, correct/total


# In[8]:


def fit(model, loaders, *,
        epochs=50, lr=1e-3, scheduler_fn=None,
        tag='?', bs=0, display_every=10):
    print(f"[{tag}] bs={bs} lr={lr:.0e} ")
    criterion  = nn.CrossEntropyLoss()
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = scheduler_fn(optimiser, epochs) if scheduler_fn else None

    history = {'train':[], 'val':[]}
    best    = {'state': None, 'val_acc': 0.0}

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = epoch_loop(model, loaders['train'], criterion, optimiser)
        v_loss,  v_acc  = epoch_loop(model, loaders['val'],   criterion)

        if scheduler: scheduler.step()
        history['train'].append({'loss':tr_loss, 'acc':tr_acc})
        history['val'  ].append({'loss':v_loss,  'acc':v_acc})

        if v_acc > best['val_acc']:
            best.update(state=copy.deepcopy(model.state_dict()), val_acc=v_acc)

        # ---------- logging ----------
        if ep % display_every == 0 or ep == epochs:
            print(
                  f"| epoch {ep:02}/{epochs}"
                  f" | train {tr_acc:.3%}/{tr_loss:.3f}"
                  f" | val {v_acc:.3%}/{v_loss:.3f}")

    model.load_state_dict(best['state'])
    return history, best['val_acc']


# # 4. Hyper-parameter Grid Search (Plain **vs** Residual)
# 

# In[9]:


def run_grid(ModelCls, *, tag,
             lrs=(1e-3, 3e-4, 1e-4), bss=(32, 64),
             epochs=50, p_drop=0.0):
    results = []
    for lr, bs in itertools.product(lrs, bss):
        loaders, _ = get_loaders(batch_size=bs)
        model      = ModelCls(p_drop=p_drop).to(DEVICE)
        print(f">> {tag.upper()}  bs={bs} lr={lr:.0e}  "
          f"params={count_params(model):.2f}M")
        sched = lambda opt, e: torch.optim.lr_scheduler.CosineAnnealingLR(opt, e)
        hist, val_acc = fit(model, loaders,
                            epochs=epochs, lr=lr, scheduler_fn=sched,
                            tag=tag, bs=bs)

        results.append({
            'model'   : tag,               # plain / res
            'bs'      : bs,
            'lr'      : lr,
            'val_acc' : val_acc,
            'history' : hist,              # kept in-memory
            'state'   : copy.deepcopy(model.state_dict())
        })

        # tidy-up GPU & RAM between runs
        del model, loaders
        torch.cuda.empty_cache();  gc.collect()

    return results


# In[10]:


plain_grid = run_grid(PlainCNN5, tag='plain')
res_grid   = run_grid(ResCNN5,   tag='res')


# In[11]:


def print_best_models(plain_grid, res_grid):
    def _test_acc(state, ModelCls, bs):
        loaders, _ = get_loaders(batch_size=bs)
        model = ModelCls().to(DEVICE)
        model.load_state_dict(state); model.eval()
        _, test_acc = epoch_loop(model, loaders['test'], nn.CrossEntropyLoss())
        return test_acc

    best_plain = max(plain_grid, key=lambda r: r['val_acc'])
    best_res   = max(res_grid,   key=lambda r: r['val_acc'])

    best_plain['test_acc'] = _test_acc(best_plain['state'], PlainCNN5, best_plain['bs'])
    best_res  ['test_acc'] = _test_acc(best_res  ['state'], ResCNN5,   best_res  ['bs'])

    for entry in (best_plain, best_res):
        print(f"\n=== BEST {entry['model'].upper()} MODEL ===")
        print(f"val acc : {entry['val_acc']:.3%}")
        print(f"test acc: {entry['test_acc']:.3%}")
        print(f"lr      : {entry['lr']:.0e}")
        print(f"batch   : {entry['bs']}")
        print("="*32)


# In[12]:


print_best_models(plain_grid, res_grid)


# In[23]:


# ---------------------------------------------------------------------------
# Confusion matrices for the best  models
# ---------------------------------------------------------------------------

# pick the highest-validation-accuracy entry from each original grid
best_plain_nd = max(plain_grid, key=lambda d: d['val_acc'])
best_res_nd   = max(res_grid,   key=lambda d: d['val_acc'])

# retrieve the batch size and state_dict that were saved
bs_plain_nd = best_plain_nd['bs']
bs_res_nd   = best_res_nd['bs']

# visualise the confusion matrices
show_confusion(best_plain_nd['state'],
               ModelCls=PlainCNN5,
               bs=bs_plain_nd,
               title=(f"Plain CNN (no dropout) | "
                      f"bs={bs_plain_nd} lr={best_plain_nd['lr']:.0e}"))

show_confusion(best_res_nd['state'],
               ModelCls=ResCNN5,
               bs=bs_res_nd,
               title=(f"Residual CNN (no dropout) | "
                      f"bs={bs_res_nd} lr={best_res_nd['lr']:.0e}"))


# # 5. Plotting Helper
# 

# In[13]:


def plot_metric_split(result_list, metric, base_title):
    fig_tr,  ax_tr  = plt.subplots(figsize=(7, 5))
    fig_val, ax_val = plt.subplots(figsize=(7, 5))

    for r in result_list:
        epochs = range(1, len(r['history']['train']) + 1)
        label  = f"bs={r['bs']}, lr={r['lr']:.0e}"

        y_tr = [e[metric] for e in r['history']['train']]
        ax_tr.plot(epochs, y_tr, '-',  label=label)

        y_val = [e[metric] for e in r['history']['val']]
        ax_val.plot(epochs, y_val, '-', label=label)

    for ax, phase in ((ax_tr, 'Train'), (ax_val, 'Validation')):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{base_title} – {phase}")
        ax.grid(alpha=.3)
        ax.legend(ncol=2, fontsize=8)

    return fig_tr, fig_val


# In[14]:


plot_metric_split(plain_grid, 'loss', 'Plain CNN Loss')
plot_metric_split(plain_grid, 'acc',  'Plain CNN Accuracy')


plot_metric_split(res_grid,   'loss', 'Residual CNN Loss')
plot_metric_split(res_grid,   'acc',  'Residual CNN Accuracy')

plt.show()          # show all open figures at once (optional)


# # 6. Dropout Re-tuning on the Best Hyper-params
# 

# In[15]:


def best_of(grid):
    return max(grid, key=lambda r: r['val_acc'])


def retune_dropout(base_entry, ModelCls, tag,
                   dropouts=(0.2, 0.4), epochs=50):

    loaders, _ = get_loaders(batch_size=base_entry['bs'])
    sched_fn   = lambda opt,e: torch.optim.lr_scheduler.CosineAnnealingLR(opt, e)

    tuned = []
    for p in dropouts:
        model = ModelCls(p_drop=p).to(DEVICE)
        print(f">> {tag.upper()}  dropout={p}  "
          f"params={count_params(model):.2f}M")
        hist, val_acc = fit(model, loaders,
                            epochs=epochs, lr=base_entry['lr'],
                            scheduler_fn=sched_fn,
                            tag=tag, bs=base_entry['bs'])

        _, test_acc = epoch_loop(model, loaders['test'],
                                 nn.CrossEntropyLoss())

        tuned.append({
            'model'   : tag,
            'bs'      : base_entry['bs'],
            'lr'      : base_entry['lr'],
            'p'       : p,
            'val_acc' : val_acc,
            'test_acc': test_acc,
            'history' : hist,
            'state'   : copy.deepcopy(model.state_dict())
        })

        print(f"dropout={p}: val {val_acc:.3%} | test {test_acc:.3%}")

        del model
        torch.cuda.empty_cache(); gc.collect()

    return tuned


# In[16]:


plain_dropout = retune_dropout(best_of(plain_grid), PlainCNN5, tag='plain')
res_dropout   = retune_dropout(best_of(res_grid),   ResCNN5, tag='res')


# # 7. Confusion Matrix for the Overall Best Model
# 

# In[17]:


def show_confusion(best_state, ModelCls, *,
                   bs=64, title='Confusion'):

    loaders, class_names = get_loaders(batch_size=bs)

    model = ModelCls().to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loaders['test']:
            y_true.extend(y.numpy())
            y_pred.extend(model(x.to(DEVICE)).argmax(1)
                          .cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(
        cm, display_labels=class_names
    ).plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title(title)
    plt.show()



# In[18]:


# ---------------------------------------------------------------------------
# Confusion matrices for the best dropout-tuned models (plain & residual)
# ---------------------------------------------------------------------------

# ❶ pick the highest-val-acc run from each family
best_plain = max(plain_dropout, key=lambda d: d['val_acc'])
best_res   = max(res_dropout,   key=lambda d: d['val_acc'])

# ❷ read stored meta-data directly
bs_plain, lr_plain = best_plain['bs'], best_plain['lr']
bs_res,   lr_res   = best_res  ['bs'], best_res  ['lr']

# ❸ visualise
show_confusion(best_plain['state'],
               ModelCls=PlainCNN5,
               bs=bs_plain,
               title=(f"Plain CNN  |  bs={bs_plain}  lr={lr_plain:.0e}  "
                      f"p={best_plain['p']}"))

show_confusion(best_res['state'],
               ModelCls=ResCNN5,
               bs=bs_res,
               title=(f"Residual CNN  |  bs={bs_res}  lr={lr_res:.0e}  "
                      f"p={best_res['p']}"))


# In[19]:


# ---------------------------------------------------------------------------
# Best-of-all selection  (grid + dropout)   +  confusion matrix
# ---------------------------------------------------------------------------

def ensure_test_acc(entry, ModelCls):
    """Add test_acc to entry if absent."""
    if 'test_acc' in entry:
        return
    loaders, _ = get_loaders(batch_size=entry['bs'])
    model = ModelCls().to(DEVICE)
    model.load_state_dict(entry['state'])
    model.eval()
    _, test_acc = epoch_loop(model, loaders['test'], nn.CrossEntropyLoss())
    entry['test_acc'] = test_acc
    del model; torch.cuda.empty_cache(); gc.collect()

def collect_all():
    """Return a single list with unified meta-data from every experiment."""
    all_runs = []

    # ①  original grid results
    for d in plain_grid + res_grid:
        d = d.copy()
        d.setdefault('p', None)        # no dropout
        d.setdefault('model', d['model'])
        d['bs'] = d['bs']
        ensure_test_acc(d, PlainCNN5 if d['model']=='plain' else ResCNN5)
        all_runs.append(d)

    # ②  dropout-tuned results (already have test_acc)
    for d in plain_dropout + res_dropout:
        all_runs.append(d)

    return all_runs

all_runs = collect_all()
best_run = max(all_runs, key=lambda d: d['test_acc'])

# ----- summary print ---------------------------------------------------------
dropout_flag = f"p={best_run['p']}" if best_run['p'] is not None else "no-dropout"
print("\n=== OVERALL BEST MODEL ===")
print(f"model   : {best_run['model']}")
print(f"val acc : {best_run['val_acc']:.3%}")
print(f"test acc: {best_run['test_acc']:.3%}")
print(f"bs      : {best_run['bs']}")
print(f"lr      : {best_run['lr']:.0e}")
print(f"dropout : {dropout_flag}")
print("="*34)

# ----- confusion matrix ------------------------------------------------------
ModelCls = (PlainCNN5 if best_run['model']=='plain'
            else ResCNN5)

show_confusion(best_run['state'],
               ModelCls=ModelCls,
               bs=best_run['bs'],
               title=(f"{best_run['model'].capitalize()} "
                      f"| bs={best_run['bs']} lr={best_run['lr']:.0e} "
                      f"| {dropout_flag}"))


# # PART 2
# ---

# In[20]:


from torchvision import models

def mobilenet_v2_ft(num_classes=11):
    """Return a MobileNetV2 with a fresh classifier (fc)."""
    net = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
    in_features = net.classifier[-1].in_features          # 1280
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net.to(DEVICE)

def freeze_until_last_fc(net):
    """Freeze *everything* except the final Linear layer."""
    for p in net.parameters(): p.requires_grad = False
    for p in net.classifier[-1].parameters(): p.requires_grad = True
    return net

def freeze_until_last2_blocks(net):
    """
    Unfreeze the last *two* inverted-residual blocks + final FC.
    MobileNetV2 features are stored in net.features (18 blocks in PyTorch impl).
    """
    # Get indices of the last two blocks
    trainable_idx = [-1, -2]       # last two blocks
    for idx, block in enumerate(net.features):
        for p in block.parameters():
            p.requires_grad = idx in trainable_idx
    # always train the FC
    for p in net.classifier[-1].parameters(): p.requires_grad = True
    return net


# In[21]:


def run_tl_grid(setup_fn, tag,
                lrs=(1e-3, 3e-4, 1e-4), bss=(32, 64),
                epochs=50, wd=1e-4):
    """
    Train MobileNetV2 for every (lr, bs) combination under a freeze policy.

    Returns
    -------
    list of dicts with keys:
        'tag', 'bs', 'lr', 'val_acc', 'test_acc', 'state', 'history'
    """
    results = []
    for lr, bs in itertools.product(lrs, bss):
        loaders, _ = get_loaders(batch_size=bs)

        net = mobilenet_v2_ft().to(DEVICE)   # fresh backbone + new head
        net = setup_fn(net)                  # freeze / unfreeze policy
        print(f">> {tag.upper()}  bs={bs} lr={lr:.0e}  "
          f"trainable={sum(p.requires_grad for p in net.parameters())} / "
          f"{count_params(net):.2f}M total")
        # only params with requires_grad = True get optimised
        optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr, weight_decay=wd
        )
        scheduler = lambda opt,e: torch.optim.lr_scheduler.CosineAnnealingLR(opt, e)

        hist, val_acc = fit(net, loaders,
                            epochs=epochs, lr=lr, scheduler_fn=scheduler,
                            tag=f"{tag}", bs=bs)

        _, test_acc = epoch_loop(net, loaders['test'], nn.CrossEntropyLoss())

        results.append(dict(tag=tag, bs=bs, lr=lr,
                            val_acc=val_acc, test_acc=test_acc,
                            history=hist,
                            state=copy.deepcopy(net.state_dict())))

        del net, loaders
        torch.cuda.empty_cache(); gc.collect()
    return results


# ---- run grids --------------------------------------------------------------
fc_grid   = run_tl_grid(freeze_until_last_fc,        tag='mobilenet_fc')
last2_grid= run_tl_grid(freeze_until_last2_blocks,   tag='mobilenet_last2')

# ---- pick the winners -------------------------------------------------------
best_fc   = max(fc_grid,   key=lambda d: d['val_acc'])
best_last = max(last2_grid,key=lambda d: d['val_acc'])

def _print_entry(e):
    print(f"\n=== {e['tag']} ===")
    print(f"val acc : {e['val_acc']:.3%}")
    print(f"test acc: {e['test_acc']:.3%}")
    print(f"lr      : {e['lr']:.0e}")
    print(f"batch   : {e['bs']}")
    print("="*32)

_print_entry(best_fc)
_print_entry(best_last)


# In[22]:


show_confusion(best_fc['state'],
               ModelCls=mobilenet_v2_ft,
               bs=best_fc['bs'],
               title=f"MobileNetV2 FC-only (lr={best_fc['lr']:.0e})")

show_confusion(best_last['state'],
               ModelCls=mobilenet_v2_ft,
               bs=best_last['bs'],
               title=f"MobileNetV2 last-2 + FC (lr={best_last['lr']:.0e})")

