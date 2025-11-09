
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
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
from Detectors.drive_dataset import DrivingOODDataset,DrivingOODDatasetNpy
from Detectors.GMM.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
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


    base_path = "/raid/007--Experiments/selforacle/training_data"
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        # Load full dataset
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(DATA_ROOT,label=0,
                                 transform=transform)
        
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


def predict(pca, gmm, dataloader):
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
    return scores, labels

def eval():
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET,ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/GMM/{dataset}"
        swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"GMM_{TYPE}_{dataset}_{attack_type}",
        config={
            "architecture": "GMM",
            "dataset": attack_data_root,
            "type": f"{TYPE}"
        }
        )
        print("Iteration: ", attack_type, dataset, attack_data_root)
        gmm = joblib.load(root+'/weights/gmm_model.pkl')
        pca = joblib.load(root+'/weights/gmm_pca.pkl')
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
        if dataset == "Udacity":
            dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel=gmm, transform=transform)
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
        
        
        all_errors, all_labels = predict(pca, gmm, dataloader)
        eval_bootstrap(all_errors, all_labels, root+"/thresholds")
        # Threshold using 95th percentile of normal sampl
        # Predict labels
        # all_results = []
        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         scores, labels = predict(pca, gmm, dataloader)
        #         print(f"Using threshold: {thr}")
        #         thr_value = joblib.load(root+f"/thresholds/{str(thr)}.pkl")
        #     #     # --- Safe AUC-ROC ---
        #         auc_roc = roc_auc_score(labels, scores)
        #         precision, recall, _ = precision_recall_curve(labels, scores)
        #         auc_pr = auc(recall, precision)
        #         preds = (scores > thr_value).astype(int)
        #         f1 = f1_score(labels, preds)
        #         conf_mat = confusion_matrix(labels, preds)
        #         tn, fp, fn, tp = conf_mat.ravel()
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

        #     print("\n=== Mean and Std Metrics Across Thresholds ===")
        #     for k, v in mean_results.items():
        #         print(f"{k}: {v:.4f}")

        #     swanlab.log(mean_results)
            


if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    #train()
    if TYPE == "eval":
        eval()
    else:
        train()
    # get_threshold()

