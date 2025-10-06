
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
import numpy as np
import torch.nn as nn
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
import time
from tqdm import tqdm
from Detectors.DeepKNN.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
DEVICE = "cuda:0"
resnet = models.resnet34(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification layer
resnet = resnet.to(DEVICE).eval()

def extract_features(model, loader, device):
    features = []
    for batch in tqdm(loader, desc="Extracting features"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device)
        with torch.no_grad():
            out = model(images)
            normed = out / out.norm(dim=1, keepdim=True)
        features.append(normed.cpu().numpy())
    return np.concatenate(features, axis=0)

def train():
    # Setup
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    base_path = "/raid/007-Xiangyu-Experiments/selforacle/training_data"
    dataset = UdacityImageDataset(base_dir=base_path, transform=transform)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    resnet = models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    
    # Feature extraction
    train_features = extract_features(resnet, train_loader, device)
    #val_features = extract_features(resnet, val_loader, device)

    k = 50
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(train_features)
    root = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepKNN"
    joblib.dump(nbrs, root+'/weights/knn_model.pkl')

    distances, _ = nbrs.kneighbors(train_features)
    kth_distances = distances[:, -1]
    threshold = np.percentile(kth_distances, 70)
    joblib.dump(threshold, root+'/thresholds/70.pkl')
    threshold = np.percentile(kth_distances, 80)
    joblib.dump(threshold, root+'/thresholds/80.pkl')
    threshold = np.percentile(kth_distances, 90)
    joblib.dump(threshold, root+'/thresholds/90.pkl')
    threshold = np.percentile(kth_distances, 95)
    joblib.dump(threshold, root+'/thresholds/95.pkl')
    threshold = np.percentile(kth_distances, 99)
    joblib.dump(threshold, root+'/thresholds/99.pkl')

    print(f"Threshold (95th percentile of kNN distances): {threshold:.4f}")



def eval():
    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"KNN_{TYPE}_{DATASET}",
        config={
            "architecture": "KNN",
            "dataset": DATASET,
            "type": TYPE
        }
        )
    
    knn_model = joblib.load(MODEL_PATH)
    resnet = models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet.to(device)

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
   
    print("Loading dataset...")
    print("Data root:", DATA_ROOT)
    print("Data type:", DATA_TYPE)
    print("Attacked data:", ATTACK_TYPE)
    dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel=knn_model, transform=transform)
    if ATTACK_TYPE == "weather":
        attacked_dataset = UdacityImageAttackDataset(base_dir=ATTACK_DATA_ROOT, data=[ATTACK_DATA_TYPE], transform=transform)

    elif ATTACK_TYPE == "attack":
        attacked_dataset = AnomalImageDataset(ATTACKED_DATA, transform=transform)
    all_labels = []
    all_errors = []
    dataset = ConcatDataset([dataset, attacked_dataset])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    tic = time.time()
    scores = []
    labels = []  # True: 1 for anomaly, 0 for normal
    tic = time.time()
    def extract_features(model, loader, device):
        features = []
        labels = []
        for batch in tqdm(loader, desc="Extracting features"):
            images, targets = batch  # Explicit unpack
            images = images.to(device)
            with torch.no_grad():
                out = model(images)
                normed = out / out.norm(dim=1, keepdim=True)
            features.append(normed.cpu().numpy())
            labels.append(targets.numpy())
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)
    features, labels = extract_features(resnet, dataloader, device)


    distances, _ = knn_model.kneighbors(features)
    scores = distances[:, -1]  # use k-th NN distance
    print(f"Evaluation time: {time.time() - tic:.2f} seconds")
    # Assume binary labels are available in val_loader or dummy ones
    
    # Evaluate
    # Predict labels
    all_results = []


    for thr in THRESHOLD_RANGE:
        print(f"Using threshold: {thr}")
        thr_value = joblib.load(f"{THRESHOLD_ROOT}/{str(thr)}.pkl")
        print(scores)
        preds = (scores > thr_value).astype(int)
        auc_roc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc_pr = auc(recall, precision)
        f1 = f1_score(labels, preds)
        conf_mat = confusion_matrix(labels, preds)
        tn, fp, fn, tp = conf_mat.ravel()
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

