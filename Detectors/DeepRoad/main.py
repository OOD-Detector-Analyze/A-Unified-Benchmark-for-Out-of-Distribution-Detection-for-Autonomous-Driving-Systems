
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
import numpy as np
import torch.nn as nn
from typing import Tuple
import torch.multiprocessing as mp
import foolbox as fb
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, ConcatDataset
from Detectors.udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset, AnomalImageDataset
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from Detectors.utils import calc_and_store_thresholds
import joblib
from torchvision import models, transforms as T
import time
from tqdm import tqdm
from Detectors.DeepRoad.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
import json
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (80, 160)
BATCH    = 32
PCA_DIM  = 50
K_TOP    = 1        # k‑NN with k=1 → minimal distance

PCA_FILE     = "pca_model.pkl"
TRAIN_NPY    = "train_vecs.npy"
THRESH_FILE  = "tau.json"

# VGG layers used by DeepRoad‑IV
CONTENT_LAYER = "16"  # conv4_2
STYLE_LAYER   = "23"  # conv5_3

_vgg = models.vgg19(weights="VGG19_Weights.DEFAULT").features.to(DEVICE).eval()

# ────────────────────────────────────────────────────────────
#  Helper: VGG content + Gram‑style embedding  (≈ original paper)
# ────────────────────────────────────────────────────────────

def _gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    x = x.view(b, c, h * w)
    return torch.matmul(x, x.transpose(1, 2)) / (c * h * w)  # (b,c,c)

@torch.no_grad()
def vgg_embed(batch: torch.Tensor) -> np.ndarray:
    feats = {}
    x = batch.to(DEVICE)
    for name, layer in _vgg._modules.items():
        x = layer(x)
        if name == CONTENT_LAYER:
            feats["content"] = x.clone()
        if name == STYLE_LAYER:
            feats["style"] = x.clone()
            break
    content = feats["content"].cpu().view(batch.size(0), -1)   # (B, C*H*W)
    style   = _gram(feats["style"]).cpu().view(batch.size(0), -1)
    return torch.cat([content, style], dim=1).numpy()           # (B, D)

# ────────────────────────────────────────────────────────────
#  Transforms & dataset loaders
# ────────────────────────────────────────────────────────────

def make_transform() -> T.Compose:
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def embed_loader(loader: DataLoader) -> Tuple[np.ndarray, list[str]]:
    vecs, names = [], []
    for imgs, fnames, *_ in tqdm(loader, desc="↳ Embedding", ncols=80):
        vecs.append(vgg_embed(imgs))
        names.append(fnames)
    return np.vstack(vecs), names


def train():
    base_dir = '/raid/007-Xiangyu-Experiments/selforacle/training_data'
    tfm  = make_transform()
    clear_ds = UdacityImageDataset(base_dir=base_dir, transform=tfm)

    # Split into train and validation sets (e.g., 80/20)
    val_fraction = 0.2
    val_size = int(len(clear_ds) * val_fraction)
    train_size = len(clear_ds) - val_size
    tr_ds, val_ds = random_split(clear_ds, [train_size, val_size])

    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    # 1️⃣  Extract embeddings
    train_vecs, _ = embed_loader(tr_loader)
    val_vecs,  _  = embed_loader(val_loader)

    # 2️⃣  PCA
    pca = PCA(n_components=PCA_DIM, whiten=False, random_state=0)
    train_pca = pca.fit_transform(train_vecs)
    val_pca   = pca.transform(val_vecs)

    # 3️⃣  minimal‑distance scores for validation set
    d_val = pairwise_distances(val_pca, train_pca, metric="euclidean")
    min_val_scores = d_val.min(axis=1)

    # 4️⃣  persist artefacts
    root = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad"
    np.save( root+"/weights/train_vecs.npy", train_pca)
    joblib.dump(pca, root+"/weights/pca_model.pkl")

    tau = float(np.percentile(min_val_scores, 70))
    with open(root+"/thresholds/70.pkl", "w") as f:
        json.dump({"tau": tau}, f)
    tau = float(np.percentile(min_val_scores, 80))
    with open(root+"/thresholds/80.pkl", "w") as f:
        json.dump({"tau": tau}, f)

    tau = float(np.percentile(min_val_scores, 90))
    with open(root+"/thresholds/90.pkl", "w") as f:
        json.dump({"tau": tau}, f)

    tau = float(np.percentile(min_val_scores, 95))
    with open(root+"/thresholds/95.pkl", "w") as f: 
        json.dump({"tau": tau}, f)

    tau = float(np.percentile(min_val_scores, 99))  
    with open(root+"/thresholds/99.pkl", "w") as f:
        json.dump({"tau": tau}, f)    

def eval():
    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"DeepRoad_{TYPE}_{DATASET}",
        config={
            "architecture": "DeepRoad",
            "dataset": DATASET,
            "type": TYPE
        }
        )
    
    base = "/raid/007-Xiangyu-Experiments/selforacle/evaluation_data"
    tfm  = make_transform()

    dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel="", transform=tfm)
    if ATTACK_TYPE == "weather":
        attacked_dataset = UdacityImageAttackDataset(base_dir=ATTACK_DATA_ROOT, data=[ATTACK_DATA_TYPE], transform=tfm)

    elif ATTACK_TYPE == "attack":
        attacked_dataset = AnomalImageDataset(ATTACKED_DATA, transform=tfm)
    
    pca       = joblib.load("/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/weights/pca_model.pkl")
    train_pca = np.load("/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/weights/train_vecs.npy")  
    dataset = ConcatDataset([dataset, attacked_dataset])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    all_min_scores = []
 
    vecs, all_labels = embed_loader(dataloader)                      # (chunk, F)
    vecs_pca = pca.transform(vecs)                      # (chunk, D)
    d_chunk = pairwise_distances(vecs_pca, train_pca, metric="euclidean")
    all_min_scores.append(d_chunk.min(axis=1))          # list of (chunk,)
    # Assume binary labels are available in val_loader or dummy ones
    # Evaluate
    # Predict labels
    all_results = []
    all_labels = np.concatenate(all_labels)
    min_scores = np.concatenate(all_min_scores)
    for thr in THRESHOLD_RANGE:
        print(f"Using threshold: {thr}")
        thr_value = json.load(open(f"{THRESHOLD_ROOT}/{str(thr)}.pkl"))['tau']
        pred_labels = (min_scores > thr_value).astype(int)

    #     # --- Safe AUC-ROC ---
        unique_classes = np.unique(all_labels)
        if len(unique_classes) < 2:
            auc_roc = np.nan
            auc_pr = np.nan
        else:
            auc_roc = roc_auc_score(all_labels, min_scores)
            precision, recall, _ = precision_recall_curve(all_labels, min_scores)
            auc_pr = auc(recall, precision)

    #     # F1 and confusion matrix still work (but may degenerate)
        f1 = f1_score(all_labels, pred_labels, zero_division=0)
        conf_mat = confusion_matrix(all_labels, pred_labels)

        # Extract TP, FP, TN, FN safely
        if conf_mat.shape == (2, 2):
            tn, fp, fn, tp = conf_mat.ravel()
        else:
            tn = fp = fn = tp = 0

        results = {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr,
            'F1': f1,
            'Threshold': float(thr_value),
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Confusion Matrix': conf_mat
        }
        print(results)
        all_results.append(results)

    # # ---- Compute mean and std (ignores NaN automatically if you use nanmean/nanstd) ----
    mean_results = {
        'AUC-ROC-mean': np.nanmean([r['AUC-ROC'] for r in all_results]),
        'AUC-PR-mean': np.nanmean([r['AUC-PR'] for r in all_results]),
        'F1-mean':     np.nanmean([r['F1'] for r in all_results]),
        'TP-mean':     np.mean([r['TP'] for r in all_results]),
        'FP-mean':     np.mean([r['FP'] for r in all_results]),
        'TN-mean':     np.mean([r['TN'] for r in all_results]),
        'FN-mean':     np.mean([r['FN'] for r in all_results]),

        'AUC-ROC-std': np.nanstd([r['AUC-ROC'] for r in all_results]),
        'AUC-PR-std':  np.nanstd([r['AUC-PR'] for r in all_results]),
        'F1-std':      np.nanstd([r['F1'] for r in all_results]),
        'TP-std':      np.std([r['TP'] for r in all_results]),
        'FP-std':      np.std([r['FP'] for r in all_results]),
        'TN-std':      np.std([r['TN'] for r in all_results]),
        'FN-std':      np.std([r['FN'] for r in all_results]),
    }

    print("\n=== Mean and Std Metrics Across Thresholds ===")
    for k, v in mean_results.items():
        print(f"{k}: {v:.4f}")

    swanlab.log(mean_results)
        

if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    #train()
    if TYPE == "eval":
        eval()
    else:
        train()
    # get_threshold()

