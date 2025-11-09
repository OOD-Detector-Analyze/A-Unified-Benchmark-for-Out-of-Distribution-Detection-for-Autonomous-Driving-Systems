import os, json, math
from pathlib import Path
import numpy as np
import joblib
from tqdm import tqdm

# ==== Your project datasets & config (unchanged imports) ====
from Detectors.drive_dataset import DrivingOODDataset, DrivingOODDatasetNpy
from Detectors.udacity_dataset import (
    UdacityImageDataset,
    UdacityImageTestDataset,
    UdacityImageAttackDataset,
    AnomalImageDataset,
)
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
from torchvision import transforms, models
from Detectors.DeepRoad.config import *  # expects: DATASET, DATA_ROOT, ATTACK_TYPE, ATTACK_DATA_ROOT, TYPE, COMMON_THRESHOLD
from torchvision import transforms as T

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix
import swanlab
from Detectors.utils import *
# =============================
# Paper-faithful DeepRoad-IV
#   VGG features -> PCA -> distance -> threshold
# =============================

ROOT_FMT = "/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/{dataset}"

def get_root(dataset: str) -> str:
    return ROOT_FMT.format(dataset=str(dataset))


class VGG19FeatureExtractor(torch.nn.Module):
    """Frozen VGG19 up to avgpool -> flattened 25088-dim feature.
    Paper references VGGNet; VGG19 keeps us close to the text.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except AttributeError:
            # compatibility for older torchvision
            vgg = models.vgg19(pretrained=True)
        self.features = vgg.features.eval()
        self.avgpool = vgg.avgpool
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)                  # (N, 512, 7, 7)
        x = torch.flatten(x, 1)              # (N, 25088)
        return x


# ---------- transforms kept consistent with your pipeline ----------

def make_transform() -> T.Compose:
    return T.Compose([
        T.Resize((160, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# ---------- helpers ----------

@torch.no_grad()
def embed_loader(dataloader: DataLoader, feat_extractor: VGG19FeatureExtractor, device: str = "cpu"):
    feats, labels = [], []
    for images, y in tqdm(dataloader, desc="Embedding"):
        images = images.to(device).float()
        f = feat_extractor(images)           # (B, 25088)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    feats = np.concatenate(feats, axis=0) if feats else np.empty((0, 25088), np.float32)
    labels = np.concatenate(labels, axis=0) if labels else np.empty((0,), np.int64)
    return feats, labels


def ensure_dirs(root: str):
    Path(os.path.join(root, "weights")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "thresholds")).mkdir(parents=True, exist_ok=True)


# =============================
# TRAIN (DeepRoad-IV)
# =============================

def train_deeproad():
    """Train paper-faithful DeepRoad-IV (PCA + nearest distance + thresholds).
    Preserves your dataset loading, split, logging, and save layout.
    """
    # --- data loaders (unchanged logic) ---
    tfm = make_transform()
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        dataset = UdacityImageDataset(base_dir=base_dir, transform=tfm)
    else:
        dataset = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=tfm)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_extractor = VGG19FeatureExtractor(device=device).eval()

    # --- collect train embeddings ---
    train_feats, _ = embed_loader(train_loader, feat_extractor, device=device)
    if train_feats.shape[0] < 2:
        raise RuntimeError("Not enough training images to fit PCA / NN.")

    # --- PCA fit (paper reduces dimension for IV). 64-256 dims are common. ---
    pca_dim = 128
    pca = PCA(n_components=pca_dim, svd_solver="randomized", whiten=False)
    Xp = pca.fit_transform(train_feats)  # (N, D)

    # --- NN index on training set in PCA space (minimal distance criterion) ---
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Xp)

    # --- compute validation distances to choose thresholds ---
    val_dists = []
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Val distances"):
            f = feat_extractor(images.to(device).float()).cpu().numpy()
            yp = pca.transform(f)
            d, _ = nn.kneighbors(yp, n_neighbors=1, return_distance=True)
            val_dists.extend(d.ravel().tolist())

    # --- standard quantile thresholds like your VAE pipeline ---
    quantiles = {"70": 0.70, "80": 0.80, "90": 0.90, "95": 0.95, "99": 0.99}
    thresholds = {name: float(np.quantile(val_dists, q)) for name, q in quantiles.items()}

    # --- save artifacts (keeps your directory layout) ---
    root = get_root(DATASET)
    ensure_dirs(root)
    joblib.dump({"pca": pca, "nn": nn}, os.path.join(root, "weights", "deeproad_iv.pkl"))
    for name, thr in thresholds.items():
        joblib.dump(thr, os.path.join(root, "thresholds", f"{name}.pkl"))

    print("✅ DeepRoad-IV training finished.")


# =============================
# EVAL (DeepRoad-IV)
# =============================

def score_pca_nn(images: torch.Tensor, fx: VGG19FeatureExtractor, pca: PCA, nn: NearestNeighbors, device: str = "cpu"):
    with torch.no_grad():
        emb = fx(images.to(device).float()).cpu().numpy()
    yp = pca.transform(emb)
    d, _ = nn.kneighbors(yp, n_neighbors=1, return_distance=True)
    return d.ravel()  # distance is the anomaly score (higher -> more OOD)


def _iter_threshold_labels(thr_spec):
    """Accept COMMON_THRESHOLD as [70,80,90] or [[70,80,90],[95,99]] etc.
    Yields strings like "70", "80" to match saved filenames.
    """
    if isinstance(thr_spec, (list, tuple, np.ndarray)):
        for item in thr_spec:
            if isinstance(item, (list, tuple, np.ndarray)):
                for sub in item:
                    yield str(int(sub))
            else:
                yield str(int(item))
    else:
        yield str(int(thr_spec))


def predict(device, dataloader: DataLoader, pca: PCA, nn: NearestNeighbors):
    feat_extractor = VGG19FeatureExtractor(device=device).eval()

    # ----- compute scores once -----
    scores, labels = [], []
    with torch.no_grad():
        for imgs, y in tqdm(dataloader, desc="Scoring"):
            s = score_pca_nn(imgs, feat_extractor, pca, nn, device)
            scores.extend(s)
            labels.extend((y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)).tolist())
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return scores,  labels  # residual and combined are the same here

def eval():
    # ATTACK_TYPE, DATASET (can be list), ATTACK_DATA_ROOT come from your config
    # Loop is preserved to match your pipeline.
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET, ATTACK_DATA_ROOT):
        root = get_root(dataset)

        swanlab.init(
            project="OODDetector",
            workspace="sumail",
            config={
                "architecture": "DeepRoad-IV (PCA + NN)",
                "dataset": str(dataset),
                "attack_root": str(attack_data_root),
                "type": f"{TYPE}",
            },
        )

        print("Iteration:", attack_type, dataset, attack_data_root)

        # ----- data -----
        tfm = make_transform()
        if dataset == "Udacity":
            base_root = DATA_ROOT
            clean_ds = UdacityImageTestDataset(base_dir=base_root, data=[DATA_TYPE], mode="clean", fmodel="", transform=tfm)
            if attack_type == "weather":
                rt, weather_type = attack_data_root
                attacked_ds = UdacityImageAttackDataset(base_dir=rt, data=[weather_type], transform=tfm)
            elif attack_type == "attack":
                attacked_ds = AnomalImageDataset(attack_data_root, transform=tfm)
            else:
                raise ValueError(f"Unknown attack_type for Udacity: {attack_type}")
        else:
            clean_ds = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=tfm)
            if attack_type == "weather":
                attacked_ds = DrivingOODDataset(base_dir=attack_data_root, label=1, transform=tfm)
            else:
                attacked_ds = DrivingOODDatasetNpy(base_dir=attack_data_root, label=1, transform=tfm)

        combo = ConcatDataset([clean_ds, attacked_ds])
        dataloader = DataLoader(combo, batch_size=32, shuffle=False, num_workers=4)

        # ----- load artifacts -----
        art_path = os.path.join(root, "weights", "deeproad_iv.pkl")
        if not os.path.exists(art_path):
            raise FileNotFoundError(f"DeepRoad-IV params not found at {art_path}; run training first.")
        art = joblib.load(art_path)
        pca, nn = art["pca"], art["nn"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scores_comb, labels = predict(device, dataloader, pca, nn)
        eval_bootstrap(scores_comb, labels, root+"/thresholds")
        # ----- per-threshold evaluation (reuse scores) -----
        # all_results = []
        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr_lbl in thr_range:
        #         thr_path = os.path.join(root, "thresholds", f"{thr_lbl}.pkl")
        #         if not os.path.exists(thr_path):
        #             print(f"[WARN] Threshold not found: {thr_path}; skipping.")
        #             continue

        #         scores_comb, labels = predict(device, dataloader, pca, nn)
        #         thr_value = float(joblib.load(thr_path))

        #         pred_labels = (scores_comb > thr_value).astype(int)  # 1 = anomaly

        #         # Metrics (safe if single-class)
        #         # Ranking metrics on combined score (recommended)
        #         if len(np.unique(labels)) < 2:
        #             auc_roc = np.nan; auc_pr = np.nan
        #         else:
        #             auc_roc = roc_auc_score(labels, scores_comb)
        #             precision, recall, _ = precision_recall_curve(labels, scores_comb)
        #             auc_pr  = auc(recall, precision)

        #         f1 = f1_score(labels, pred_labels, zero_division=0)
        #         conf_mat = confusion_matrix(labels, pred_labels)
        #         if conf_mat.shape == (2,2):
        #             tn, fp, fn, tp = conf_mat.ravel()
        #         else:
        #             tn = fp = fn = tp = 0

        #         results = {
        #             'AUC-ROC': float(auc_roc),
        #             'AUC-PR': float(auc_pr),
        #             'F1': float(f1),
        #             'Threshold(residual)': float(thr_value),
        #             'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        #             'Confusion Matrix': conf_mat
        #         }
        #         print(results); all_results.append(results)

        #     mean_results = {
        #     f'AUC-ROC-mean@{value}': np.nanmean([r['AUC-ROC'] for r in all_results]),
        #     f'AUC-PR-mean@{value}': np.nanmean([r['AUC-PR'] for r in all_results]),
        #     f'F1-mean@{value}':     np.nanmean([r['F1'] for r in all_results]),
        #     f'TP-mean@{value}':     np.mean([r['TP'] for r in all_results]),
        #     f'FP-mean@{value}':     np.mean([r['FP'] for r in all_results]),
        #     f'TN-mean@{value}':     np.mean([r['TN'] for r in all_results]),
        #     f'FN-mean@{value}':     np.mean([r['FN'] for r in all_results]),

        #     f'AUC-ROC-std{value}': np.nanstd([r['AUC-ROC'] for r in all_results]),
        #     f'AUC-PR-std{value}':  np.nanstd([r['AUC-PR'] for r in all_results]),
        #     f'F1-std{value}':      np.nanstd([r['F1'] for r in all_results]),
        #     f'TP-std{value}':      np.std([r['TP'] for r in all_results]),
        #     f'FP-std{value}':      np.std([r['FP'] for r in all_results]),
        #     f'TN-std{value}':      np.std([r['TN'] for r in all_results]),
        #     f'FN-std{value}':      np.std([r['FN'] for r in all_results]),
        #     }
        #     print("\n=== Mean and Std Metrics Across Thresholds (FAnoGAN) ===")
        #     for k, v in mean_results.items():
        #         print(f"{k}: {v:.4f}")

        #     swanlab.log(mean_results)


# =============================
# (Optional) Metamorphic Testing scaffold (DeepRoad-MT)
# You can hook this into your pipeline if/when you have UNIT and a driving model.
# =============================

@torch.no_grad()
def metamorphic_test(unit_model, driving_model, images: torch.Tensor, epsilon_deg: float, device: str = "cpu"):
    """Return per-image steering deltas and inconsistency flags.
    - unit_model: provides translate(images, target_domain)
    - driving_model: maps images -> steering angle (degrees)
    - epsilon_deg: tolerance threshold for |Δθ|
    """
    images = images.to(device)
    y_orig = driving_model(images)  # shape (B,) degrees
    rain = unit_model.translate(images, target_domain="rain")
    snow = unit_model.translate(images, target_domain="snow")

    y_rain = driving_model(rain)
    y_snow = driving_model(snow)

    d_rain = (y_rain - y_orig).abs().detach().cpu().numpy()
    d_snow = (y_snow - y_orig).abs().detach().cpu().numpy()

    inc_rain = (d_rain > epsilon_deg)
    inc_snow = (d_snow > epsilon_deg)

    return {
        "delta_rain": d_rain,
        "delta_snow": d_snow,
        "inconsistency_rain": inc_rain.astype(np.int32),
        "inconsistency_snow": inc_snow.astype(np.int32),
    }


# =============================
# main
# =============================

if __name__ == "__main__":
    # Keep your TYPE switch: train vs eval
    if TYPE == "eval":
        eval()
    else:
        train_deeproad()
