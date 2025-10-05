
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
from torch.utils.data import DataLoader, ConcatDataset
from Detectors.udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset, AnomalImageDataset
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from Detectors.utils import calc_and_store_thresholds
import joblib
import time
from Detectors.SAE.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 




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
    

def train():
    root = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/SAE"

    base_dir = '/raid/007-Xiangyu-Experiments/selforacle/training_data'
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset
    dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)

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
    weights = torch.load("/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/SAE/weights/best_sae.pth", weights_only=False)
    model.load_state_dict(weights)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    base_dir = '/raid/007-Xiangyu-Experiments/selforacle/training_data'
    dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
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
    root = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/SAE"
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




def eval():
    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"SAE_{TYPE}_{DATASET}",
        config={
            "architecture": "SAE",
            "dataset": DATASET,
            "type": TYPE
        }
        )
    model = Autoencoder().to(device)
    weights = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(weights)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
   
    print("Loading dataset...")
    print("Data root:", DATA_ROOT)
    print("Data type:", DATA_TYPE)
    print("Attacked data:", ATTACK_TYPE)
    dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel=model, transform=transform)
    if ATTACK_TYPE == "weather":
        attacked_dataset = UdacityImageAttackDataset(base_dir=ATTACK_DATA_ROOT, data=[ATTACK_DATA_TYPE], transform=transform)


    elif ATTACK_TYPE == "attack":
        transform = transforms.Compose([
            transforms.Resize((160, 320)),
            transforms.ToTensor(),

        ])
        attacked_dataset = AnomalImageDataset(ATTACKED_DATA, transform=transform)
    all_labels = []
    all_errors = []
    dataset = ConcatDataset([dataset, attacked_dataset])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    tic = time.time()
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

    
    # Threshold using 95th percentile of normal sampl
    # Predict labels
    all_results = []


    for thr in THRESHOLD_RANGE:
        print(f"Using threshold: {thr}")
        thr_value = joblib.load(f"{THRESHOLD_ROOT}/{str(thr)}.pkl")
        pred_labels = (all_errors > thr_value).astype(int)

    #     # --- Safe AUC-ROC ---
        unique_classes = np.unique(all_labels)
        if len(unique_classes) < 2:
            auc_roc = np.nan
            auc_pr = np.nan
        else:
            auc_roc = roc_auc_score(all_labels, all_errors)
            precision, recall, _ = precision_recall_curve(all_labels, all_errors)
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
    #get_threshold()

