from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torchvision.models as models
from Detectors.udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset, AnomalImageDataset
from scipy.stats import gamma
import matplotlib.pyplot as plt
from Detectors.utils import calc_and_store_thresholds
import joblib
import time
from tqdm import tqdm
from Detectors.drive_dataset import DrivingOODDataset, DrivingOODDatasetNpy
from Detectors.KDE.config import *  # reusing your config file (paths, DATASET, etc.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity  # <<< KDE
# from sklearn.mixture import GaussianMixture  # <<< removed
from Detectors.utils import *
DEVICE = "cuda:0"
resnet = models.resnet34(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification layer
resnet = resnet.to(DEVICE).eval()

def extract_features(loader):
    features = []
    for batch in tqdm(loader, desc="Extracting features"):
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images = images.to(DEVICE)
        with torch.no_grad():
            output = resnet(images)
        features.append(output.cpu().numpy())
    return np.concatenate(features, axis=0)

# keep folder layout but change model filenames to KDE
root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/KDE/{DATASET}"

def train():
    resnet = models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove the final classification layer
    resnet.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet.to(device)

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    batch_size = 32
    pca_dim = 50
    # KDE hyperparam we’ll select via a small sweep on the val set:
    bandwidth_grid = np.logspace(-1, 1, 7)  # 0.1 ... 10.0

    # ----- dataset -----
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(
            DATA_ROOT,
            label=0,
            transform=transform
        )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    # ----- features -----
    train_features = extract_features(train_loader)
    val_features   = extract_features(val_loader)

    #----- PCA -----
    pca = PCA(n_components=pca_dim, random_state=0)
    train_pca = pca.fit_transform(train_features)
    val_pca   = pca.transform(val_features)

    # ----- KDE: pick bandwidth by validation mean log-likelihood -----
    best_bw, best_ll, best_kde = None, -np.inf, None
    for bw in bandwidth_grid:
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        kde.fit(train_pca)
        # score_samples returns per-sample log density; higher is better
        val_ll = kde.score_samples(val_pca).mean() if len(val_pca) else -np.inf
        if val_ll > best_ll:
            best_ll, best_bw, best_kde = val_ll, bw, kde
    if best_kde is None:
        raise RuntimeError("KDE bandwidth sweep failed to fit any model.")

    # save model + pca
    joblib.dump(best_kde, root + '/weights/kde_model.pkl')
    joblib.dump(pca,      root + '/weights/kde_pca.pkl')

    # ----- scores & thresholds (higher score ⇒ more anomalous, so negate log-density) -----
    train_score = -best_kde.score_samples(train_pca)
    val_scores  = -best_kde.score_samples(val_pca)

    for pct, name in [(70,'70'), (80,'80'), (90,'90'), (95,'95'), (99,'99')]:
        threshold = np.percentile(train_score, pct)
        joblib.dump(threshold, root + f'/thresholds/{name}.pkl')

    print(f"[KDE] best bandwidth: {best_bw:.4f} | train mean score: {train_score.mean():.4f} | val mean score: {val_scores.mean():.4f}")


import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import joblib
from tqdm import tqdm

def refresh_thresholds(
    DATASET,
    DATA_ROOT,
    root,
    resnet,
    device,
    _extract_eval_features,
    batch_size=32,
    percentiles=(70, 80, 90, 95, 99),
    num_workers=4,
    use_saved_train_indices=True,
):
    """
    Reload saved PCA and KDE, recompute anomaly-score thresholds from the (train) data,
    and overwrite existing threshold .pkl files under {root}/thresholds/.
    """

    weights_dir = os.path.join(root, "weights")
    thr_dir     = os.path.join(root, "thresholds")
    splits_dir  = os.path.join(root, "splits")
    os.makedirs(thr_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # ---- load trained PCA + KDE ----
    kde_path = os.path.join(weights_dir, "kde_model.pkl")
    pca_path = os.path.join(weights_dir, "kde_pca.pkl")
    if not (os.path.isfile(kde_path) and os.path.isfile(pca_path)):
        raise FileNotFoundError(f"Could not find saved model files at:\n  {kde_path}\n  {pca_path}")

    best_kde = joblib.load(kde_path)
    pca      = joblib.load(pca_path)

    # ---- dataset & transform ----
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(DATA_ROOT, label=0, transform=transform)

    # ---- use saved train indices if available ----
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(0)
    train_dataset, _ = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # ---- extract features ----
    print("[KDE] Extracting features for threshold refresh...",train_loader.dataset.__len__())
    train_features, _ = _extract_eval_features(train_loader)
    if train_features.shape[0] == 0:
        raise RuntimeError("No features extracted from training set.")

    # ---- PCA transform and KDE scoring ----
    train_pca   = pca.transform(train_features)
    train_score = -best_kde.score_samples(train_pca)

    # ---- recompute and overwrite thresholds ----
    for pct in percentiles:
        thr_value = float(np.percentile(train_score, pct))
        joblib.dump(thr_value, os.path.join(thr_dir, f"{pct}.pkl"))
        print(f"[KDE] threshold {pct}% -> {thr_value:.6f}")

    print(f"[KDE] thresholds refreshed from {len(train_score)} samples.")


def _extract_eval_features(loader):
        features, all_labels = [], []
        for batch in tqdm(loader, desc="Evaluating"):
            if isinstance(batch, (tuple, list)):
                images, targets = batch
            else:
                images = batch
                targets = torch.zeros(len(images))
            images = images.to(device)
            with torch.no_grad():
                output = resnet(images)
            features.append(output.cpu().numpy())
            all_labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(all_labels)

def predict(pca, kde, dataloader, tic):
    val_features, labels = _extract_eval_features(dataloader)
    val_pca = pca.transform(val_features)
    scores = -kde.score_samples(val_pca)  # higher = more anomalous
    print(f"Evaluation time: {time.time() - tic:.2f} seconds")

    labels = np.asarray(labels).astype(int)
    return scores, labels




def eval():
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET,ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/KDE/{dataset}"
        swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"KDE{TYPE}_{dataset}_{attack_type}",
        config={
            "architecture": "KDE",
            "dataset": attack_data_root,
            "type": f"{TYPE}"
        }
        )
        print("Iteration: ", attack_type, dataset, attack_data_root)
        # load models
        kde = joblib.load(root + '/weights/kde_model.pkl')
        pca = joblib.load(root + '/weights/kde_pca.pkl')

        # backbone for feature extraction
        transform = transforms.Compose([
            transforms.Resize((160, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print("Loading dataset...")
        print("Data root:", DATA_ROOT)
        print("Data type:", ATTACK_TYPE)


        # normal + attacked
        if dataset == "Udacity":
            dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=["CHAUFFEUR-Track1-Normal"], mode="clean", fmodel="", transform=transform)
            if attack_type == "weather":
                rt, weather_type = attack_data_root
                attacked_dataset = UdacityImageAttackDataset(base_dir=rt, data=[weather_type], transform=transform)
            elif attack_type == "attack":
                attacked_dataset = AnomalImageDataset(attack_data_root, transform=transform)
        else:
            dataset = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=transform)
            if attack_type == "weather":
                attacked_dataset = DrivingOODDataset(base_dir=attack_data_root, label=1, transform=transform)
            else:
                attacked_dataset = DrivingOODDatasetNpy(base_dir=attack_data_root, label=1, transform=transform)

        dataset = ConcatDataset([dataset, attacked_dataset])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # --- feature extraction ---
        tic = time.time()
        
        all_results = []
        all_errors, all_labels = predict(pca, kde, dataloader, tic)
        eval_bootstrap(all_errors, all_labels, root+"/thresholds")
        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         all_errors, all_labels = predict(pca, kde, dataloader, tic)
        #         thr_value = float(joblib.load(root + f"/thresholds/{str(thr)}.pkl"))
        #         print(thr_value)
        #         with open("list.txt", "w") as f:
        #             for item in all_errors:
        #                 f.write(f"{item}\n")
        #         pred_labels = (all_errors > thr_value).astype(int)
        #         # Robust metrics (handle degenerate cases if only one class present)
        #         unique_classes = np.unique(all_labels)
        #         if len(unique_classes) < 2:
        #             auc_roc = np.nan
        #             auc_pr = np.nan
        #         else:
        #             auc_roc = roc_auc_score(all_labels, all_errors)
        #             precision, recall, _ = precision_recall_curve(all_labels, all_errors)
        #             auc_pr = auc(recall, precision)

        #     #     # F1 and confusion matrix still work (but may degenerate)
        #         f1 = f1_score(all_labels, pred_labels, zero_division=0)
        #         conf_mat = confusion_matrix(all_labels, pred_labels)

        #         # Extract TP, FP, TN, FN safely
        #         if conf_mat.shape == (2, 2):
        #             tn, fp, fn, tp = conf_mat.ravel()
        #         else:
        #             tn = fp = fn = tp = 0

        #         results = {
        #             'AUC-ROC': auc_roc,
        #             'AUC-PR': auc_pr,
        #             'F1': f1,
        #             'Threshold': float(thr_value),
        #             'TP': tp,
        #             'FP': fp,
        #             'TN': tn,
        #             'FN': fn,
        #             'Confusion Matrix': conf_mat
        #         }
        #         print(results)
        #         all_results.append(results)

        #     # # ---- Compute mean and std (ignores NaN automatically if you use nanmean/nanstd) ----
        #     mean_results = {
        #         f'AUC-ROC-mean@{value}': np.nanmean([r['AUC-ROC'] for r in all_results]),
        #         f'AUC-PR-mean@{value}': np.nanmean([r['AUC-PR'] for r in all_results]),
        #         f'F1-mean@{value}':     np.nanmean([r['F1'] for r in all_results]),
        #         f'TP-mean@{value}':     np.mean([r['TP'] for r in all_results]),
        #         f'FP-mean@{value}':     np.mean([r['FP'] for r in all_results]),
        #         f'TN-mean@{value}':     np.mean([r['TN'] for r in all_results]),
        #         f'FN-mean@{value}':     np.mean([r['FN'] for r in all_results]),

        #         f'AUC-ROC-std{value}': np.nanstd([r['AUC-ROC'] for r in all_results]),
        #         f'AUC-PR-std{value}':  np.nanstd([r['AUC-PR'] for r in all_results]),
        #         f'F1-std{value}':      np.nanstd([r['F1'] for r in all_results]),
        #         f'TP-std{value}':      np.std([r['TP'] for r in all_results]),
        #         f'FP-std{value}':      np.std([r['FP'] for r in all_results]),
        #         f'TN-std{value}':      np.std([r['TN'] for r in all_results]),
        #         f'FN-std{value}':      np.std([r['FN'] for r in all_results]),
        #     }

        #     print("\n=== Mean and Std Metrics Across Thresholds ===")
        #     for k, v in mean_results.items():
        #         print(f"{k}: {v:.4f}")

        #     swanlab.log(mean_results)
            

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    if TYPE == "eval":
        eval()
    else:
        train()
