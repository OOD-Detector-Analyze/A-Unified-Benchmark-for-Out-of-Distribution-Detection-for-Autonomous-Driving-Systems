
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
from Detectors.GMM.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
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

def train():
    resnet = models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove the final classification layer
    resnet.eval()  # Set the model to evaluation mode

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    batch_size = 32
    pca_dim = 50
    n_components = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet.to(device)


    base_path = "/raid/007-Xiangyu-Experiments/selforacle/training_data"
    dataset = UdacityImageDataset(base_dir=base_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_features = extract_features(train_loader)
    val_features = extract_features(val_loader)
    pca = PCA(n_components=pca_dim)
    train_pca = pca.fit_transform(train_features)
    val_pca = pca.transform(val_features)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(train_pca)
    root = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/GMM"
    joblib.dump(gmm, root+'/weights/gmm_model.pkl')
    joblib.dump(pca, root+'/weights/gmm_pca.pkl')

    train_score = -gmm.score_samples(train_pca)
    val_scores = -gmm.score_samples(val_pca)
    threshold = np.percentile(train_score, 70)
    joblib.dump(threshold, root+'/thresholds/70.pkl')
    threshold = np.percentile(train_score, 80)
    joblib.dump(threshold, root+'/thresholds/80.pkl')
    threshold = np.percentile(train_score, 90)
    joblib.dump(threshold, root+'/thresholds/90.pkl')
    threshold = np.percentile(train_score, 95)
    joblib.dump(threshold, root+'/thresholds/95.pkl')
    threshold = np.percentile(train_score, 99)
    joblib.dump(threshold, root+'/thresholds/99.pkl')
    print(f"train loss: {train_score}   val loss {val_scores}")



def eval():
    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"GMM_{TYPE}_{DATASET}",
        config={
            "architecture": "GMM",
            "dataset": DATASET,
            "type": TYPE
        }
        )
    
    gmm = joblib.load('/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/GMM/weights/gmm_model.pkl')
    pca = joblib.load('/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/GMM/weights/gmm_pca.pkl')
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
    dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel=gmm, transform=transform)
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
    def extract_features(loader):
        features, all_labels = [], []
        for batch in tqdm(loader, desc="Evaluating"):
            if isinstance(batch, (tuple, list)):
                images, targets = batch
            else:
                images = batch
                targets = torch.zeros(len(images))  # Dummy labels if not available
            images = images.to(device)
            with torch.no_grad():
                output = resnet(images)
            features.append(output.cpu().numpy())
            all_labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(all_labels)



    val_features, labels = extract_features(dataloader)
    val_pca = pca.transform(val_features)
    val_scores = -gmm.score_samples(val_pca)  # higher = more anomalous
    print(f"Evaluation time: {time.time() - tic:.2f} seconds")
    # Estimate threshold again using validation set (optional: load from training)

    # Binary classification
    scores = np.array(val_scores)
    labels = np.array(labels)

    
    # Threshold using 95th percentile of normal sampl
    # Predict labels
    all_results = []


    for thr in THRESHOLD_RANGE:
        print(f"Using threshold: {thr}")
        thr_value = joblib.load(f"{THRESHOLD_ROOT}/{str(thr)}.pkl")
    #     # --- Safe AUC-ROC ---
        auc_roc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc_pr = auc(recall, precision)
        preds = (scores > thr_value).astype(int)
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

