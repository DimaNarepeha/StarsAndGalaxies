
import os, math, itertools, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from Model import Model

# ========= USER SETTINGS (adjust paths if needed) =========
CKPT_PATH   = "lightning_logs/version_2/checkpoints/88.7%epoch=32-step=8250.ckpt"
DATA_VAL    = "data/validate"
DATA_TEST   = "data/test"           # can switch to test below
OUT_DIR     = "thesis_figs"
BATCH_SIZE  = 64
NUM_WORKERS = min(8, os.cpu_count() or 2)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
SEED        = 42

# ========= UTILS =========
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def denorm_to_img(t):  # t: (3,H,W) on CPU
    x = t.numpy().transpose(1,2,0)
    x = (x * np.array(STD)) + np.array(MEAN)
    return np.clip(x, 0, 1)

def savefig_tight(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ========= DATA =========
def make_loader(root, batch_size=BATCH_SIZE):
    tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    ds  = datasets.ImageFolder(root=root, transform=tf)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return ds, dl

# ========= EVAL HELPERS =========
def collect_preds(model, loader, device=DEVICE):
    model.eval()
    all_probs, all_preds, all_labels, all_imgs = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_imgs.extend([img.cpu() for img in imgs])
    probs  = torch.cat(all_probs).numpy()
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return probs, preds, labels, all_imgs

def save_prediction_grid(imgs, labels, preds, probs, class_names, which="correct", k=8, path="grid.png"):
    idx = np.where(preds == labels)[0] if which == "correct" else np.where(preds != labels)[0]
    if len(idx) == 0:
        print(f"[warn] No {which} examples found.")
        return
    # choose k highest-confidence
    conf = probs[idx, preds[idx]]
    sel = idx[np.argsort(-conf)[:k]]

    cols = 4
    rows = math.ceil(len(sel)/cols)
    plt.figure(figsize=(4*cols, 3*rows))
    for i, j in enumerate(sel):
        plt.subplot(rows, cols, i+1)
        plt.imshow(denorm_to_img(imgs[j]))
        title = f"T:{class_names[labels[j]]}  P:{class_names[preds[j]]}\nconf={probs[j, preds[j]]:.2f}"
        plt.title(title, fontsize=9)
        plt.axis("off")
    savefig_tight(path)

def plot_confusion(cm, classes, normalize=False, title="Confusion Matrix", path="confusion.png"):
    if normalize:
        cm = cm.astype('float') / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.0 if cm.size else 0.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        plt.text(j, i, format(val, fmt),
                 ha="center",
                 color="white" if val > thresh else "black",
                 fontsize=9)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    savefig_tight(path)

def save_saliency_overlay(model, imgs, labels, preds, class_names, k=6, path="fig_saliency_overlay.png"):
    idx_correct = np.where(preds == labels)[0][:k//2]
    idx_wrong   = np.where(preds != labels)[0][:k//2]
    sel = np.concatenate([idx_correct, idx_wrong]) if len(idx_wrong)>0 else idx_correct
    if len(sel) == 0:
        print("[warn] Not enough samples for saliency.")
        return

    def saliency_map(m, img_tensor):
        m.eval()
        img = img_tensor.unsqueeze(0).to(DEVICE).clone().detach().requires_grad_(True)
        logits = m(img)
        target = logits.argmax(dim=1)
        loss = logits[0, target]
        loss.backward()
        sal = img.grad.detach().abs().squeeze(0).sum(dim=0)
        sal = sal.cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal

    cols = 3
    rows = math.ceil(len(sel)/cols)
    plt.figure(figsize=(4*cols, 3.2*rows))
    for i, j in enumerate(sel):
        img = imgs[j]
        sal = saliency_map(model, img)
        rgb = denorm_to_img(img)
        plt.subplot(rows, cols, i+1)
        plt.imshow(rgb, alpha=0.85)
        plt.imshow(sal, cmap='jet', alpha=0.35)
        plt.axis('off')
        plt.title(f"T:{class_names[labels[j]]}  P:{class_names[preds[j]]}", fontsize=9)
    savefig_tight(os.path.join(OUT_DIR, path))

def save_weight_histograms(model, threshold=1.0):
    # Saves weight histograms (like your plot function) into OUT_DIR
    for name, p in model.named_parameters():
        if 'weight' not in name:
            continue
        w = p.data.detach().cpu().numpy().flatten()
        large = w[w > threshold]
        plt.figure(figsize=(6,4))
        plt.hist(w, bins=50, alpha=0.7, label='All weights')
        if large.size > 0:
            plt.hist(large, bins=50, alpha=0.6, label=f'> {threshold}')
        plt.title(f'Weight Distribution: {name}')
        plt.xlabel('Weight value'); plt.ylabel('Frequency')
        plt.legend()
        savefig_tight(os.path.join(OUT_DIR, f'weights_{name.replace(".","_")}.png'))

# ========= MAIN =========
def main():
    set_seed(SEED)
    ensure_out()

    # Load data (choose VAL; switch to TEST if you prefer)
    ds, loader = make_loader(DATA_VAL)
    class_names = ds.classes
    print(f"[info] Classes: {class_names}")

    # Load model from your checkpoint
    model = Model.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        batch_size=BATCH_SIZE,
        learning_rate=0.001,
        num_classes=len(class_names),
        epsilon=1e-8
    ).to(DEVICE)

    # Collect predictions
    probs, preds, labels, imgs = collect_preds(model, loader)

    # 1) Grids of predictions
    save_prediction_grid(imgs, labels, preds, probs, class_names, "correct",
                         8, os.path.join(OUT_DIR,"fig_predictions_correct.png"))
    save_prediction_grid(imgs, labels, preds, probs, class_names, "incorrect",
                         8, os.path.join(OUT_DIR,"fig_predictions_misclassified.png"))

    # 2) Confusion matrices + per-class metrics CSV
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    plot_confusion(cm, class_names, normalize=False,
                   title="Confusion Matrix",
                   path=os.path.join(OUT_DIR,"fig_confusion_counts.png"))
    plot_confusion(cm, class_names, normalize=True,
                   title="Confusion Matrix (normalized)",
                   path=os.path.join(OUT_DIR,"fig_confusion_norm.png"))

    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    import pandas as pd
    pd.DataFrame(report).T.to_csv(os.path.join(OUT_DIR,"table_per_class_metrics.csv"), index=True)

    # 3) ROC & PR (only if binary)
    if len(class_names) == 2:
        # choose positive class (use 'galaxies' if present)
        pos_name = 'galaxies' if 'galaxies' in class_names else class_names[1]
        pos_idx  = class_names.index(pos_name)
        y_true   = (labels == pos_idx).astype(int)
        y_score  = probs[:, pos_idx]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC — {pos_name} vs other'); plt.legend()
        savefig_tight(os.path.join(OUT_DIR,"fig_roc.png"))

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Precision–Recall — {pos_name} vs other')
        savefig_tight(os.path.join(OUT_DIR,"fig_pr.png"))
    else:
        print("[info] ROC/PR skipped (not binary).")

    # 4) Saliency overlays (simple gradients)
    save_saliency_overlay(model, imgs, labels, preds, class_names, k=6,
                          path="fig_saliency_overlay.png")

    # 5) Optional: weight histograms saved (no interactive show)
    save_weight_histograms(model, threshold=1.0)

    # 6) Captions helper
    captions = {
        "fig_predictions_correct.png": "Correct predictions with confidence.",
        "fig_predictions_misclassified.png": "Misclassified examples; typical faint/compact confusions.",
        "fig_confusion_counts.png": "Confusion matrix (counts). Rows=true; cols=pred.",
        "fig_confusion_norm.png": "Confusion matrix (normalized per true class).",
        "fig_roc.png": "ROC curve (binary).",
        "fig_pr.png": "Precision–Recall curve (binary).",
        "fig_saliency_overlay.png": "Saliency overlays (red = strongest contribution).",
        "table_per_class_metrics.csv": "Per-class precision, recall, F1, and accuracy."
    }
    with open(os.path.join(OUT_DIR, "README_captions.json"), "w") as f:
        json.dump(captions, f, indent=2)

    print(f"[done] Saved figures to: {OUT_DIR}")

if __name__ == "__main__":
    main()
