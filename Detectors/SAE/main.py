
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from Detectors.udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset, AnomalImageDataset
import numpy as np
from scipy.stats import gamma

import matplotlib.pyplot as plt
from Detectors.utils import calc_and_store_thresholds
from Detectors.utils import *
import joblib
import time
from Detectors.SAE.config import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from Detectors.drive_dataset import DrivingDataset, DrivingOODDataset, DrivingOODDatasetNpy



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 80, 160]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 32, 40, 80]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 20, 40]
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # [B, 32, 40, 80]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 80, 160]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # [B, 3, 160, 320]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/SAE/{DATASET}"
def train():
    transform = transforms.Compose([
            transforms.Resize((160, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'

        # Load full dataset
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(DATA_ROOT,label=0,
                                 transform=transform)
        
        # Split into train and validation sets (e.g., 80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    best_val_loss = float('inf')

    best_val_loss = float('inf')
    train_losses_for_threshold = []

    for epoch in range(num_epochs):
        # -------- Training --------
        model.train()
        train_loss = 0.0
        train_losses_for_threshold.clear()

        for images, label in train_loader:
            images = images.to(device)
            outputs = model(images)

            # Standard training loss
            loss = criterion(outputs, images)

            # Per-sample loss for threshold (disable grad)
            with torch.no_grad():
                model.eval()
                losses = torch.nn.functional.mse_loss(outputs, images, reduction='none')
                losses = losses.view(losses.size(0), -1).mean(dim=1)
                train_losses_for_threshold.extend(losses.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, label in val_loader:

                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # -------- Save Best Model and Threshold --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), root+"/weights/best_sae.pth")
            print(f"✅ Saved best model at epoch {epoch+1} with val loss {val_loss:.4f}")

            threshold_dict = calc_and_store_thresholds(train_losses_for_threshold, "SAE")
            threshold = threshold_dict["0.7"]
            joblib.dump(threshold, root+"/thresholds/70.pkl")
            threshold = threshold_dict["0.8"]
            joblib.dump(threshold, root+"/thresholds/80.pkl")
            threshold = threshold_dict["0.9"]
            joblib.dump(threshold, root+"/thresholds/90.pkl")
            threshold = threshold_dict["0.95"]
            joblib.dump(threshold, root+"/thresholds/95.pkl")
            threshold = threshold_dict["0.99"]
            joblib.dump(threshold, root+"/thresholds/99.pkl")


def get_threshold():
    model = Autoencoder().to(device)
    weights = torch.load(root+"/weights/best_sae.pth", weights_only=False)
    model.load_state_dict(weights)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    base_dir = '/raid/007--Experiments/selforacle/training_data'
    dataset = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    train_losses_for_threshold = []
    train_loss = 0.0
    for images, label in train_loader:
        images = images.to(device)
        outputs = model(images)
        # Standard training loss
        loss = criterion(outputs, images)
        # Per-sample loss for threshold (disable grad)
        with torch.no_grad():
            losses = torch.nn.functional.mse_loss(outputs, images, reduction='none')
            losses = losses.view(losses.size(0), -1).mean(dim=1)
            train_losses_for_threshold.extend(losses.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    threshold_dict = calc_and_store_thresholds(train_losses_for_threshold, "SAE")
    threshold = threshold_dict["0.7"]
    joblib.dump(threshold, root+"/thresholds/70.pkl")
    threshold = threshold_dict["0.8"]
    joblib.dump(threshold, root+"/thresholds/80.pkl")
    threshold = threshold_dict["0.9"]
    joblib.dump(threshold, root+"/thresholds/90.pkl")
    threshold = threshold_dict["0.95"]
    joblib.dump(threshold, root+"/thresholds/95.pkl")
    threshold = threshold_dict["0.99"]
    joblib.dump(threshold, root+"/thresholds/99.pkl")

def predict(dataloader, model, tic ):
    all_errors = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            x_recon = model(x)

            # Calculate reconstruction error per sample
            errors = F.mse_loss(x_recon, x, reduction='none')
            errors = errors.view(errors.size(0), -1).mean(dim=1)  # mean error per sample

            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    print("Evaluation Time ", time.time() - tic)
    # Stack all

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    return all_errors, all_labels






def eval():
    print(DATASET)
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET, ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/SAE/{dataset}"
        swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"SAE_{TYPE}_{dataset}_{attack_type}",
        
        config={
            "architecture": "SAE",
            "dataset": attack_data_root,
            "type": f"{TYPE}"
        }
        )
        print("Iteration: ", attack_type, dataset, attack_data_root)
        model = Autoencoder().to(device)
        weights = torch.load(root+"/weights/best_sae.pth", weights_only=False)
        model.load_state_dict(weights)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((160, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        print("Loading dataset...")
        if dataset == "Udacity":
            dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel=model, transform=transform)
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

        tic = time.time()

        all_errors, all_labels = predict(dataloader,model,tic)

        eval_bootstrap(all_errors, all_labels, root+"/thresholds")

        # np.savetxt("results.txt", all_errors, fmt="%.6f")
        # np.savetxt("results_labels.txt", all_labels, fmt="%.6f")
        # Threshold using 95th percentile of normal sampl
        # Predict labels
        # all_results = []
        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         print(f"Using threshold: {thr}")
        #         all_errors, all_labels = predict(dataloader,model,tic)
        #         thr_value = joblib.load(root+f"/thresholds/{str(thr)}.pkl")
        #         pred_labels = (all_errors > thr_value).astype(int)

        #     #     # --- Safe AUC-ROC ---
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
    #get_threshold()

