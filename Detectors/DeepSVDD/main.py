
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
from Detectors.drive_dataset import DrivingOODDataset, DrivingOODDatasetNpy
import time
from tqdm import tqdm
from Detectors.DeepSVDD.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from Detectors.utils import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
DEVICE = "cuda:0"
resnet = models.resnet34(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification layer
resnet = resnet.to(DEVICE).eval()

class CIFAR10_LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        # Corrected FC layer to match output size
        self.fc1 = nn.Linear(128 * 20 * 40, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
def init_center_c(train_loader: DataLoader, net, device, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data, _ in train_loader:
            
            # get the inputs of the batch
            inputs = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepSVDD/{DATASET}"
def train():
    base_dir = '/raid/007--Experiments/selforacle/training_data'
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset
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
 
    model = CIFAR10_LeNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-7,
                               amsgrad=True)
    num_epochs = 50
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)
    c = init_center_c(train_loader, model, DEVICE)
    model.train()

    for epoch in range(num_epochs):
        all_train_distances = []
        scheduler.step()
        train_loss = 0.0

        n_batches = 0
        for images,_ in train_loader:
            images = images.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            dist = torch.sum((outputs - c) ** 2, dim=1)
            all_train_distances.extend(dist.cpu().detach().numpy())
            loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation loop
        model.eval()
        val_loss = 0.0
        for images, _ in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            loss = torch.mean(dist)
            val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), root+"/weights/best_svdd.pth")
            all_train_distances = np.array(all_train_distances)
            threshold = np.percentile(all_train_distances, 70)
            joblib.dump(threshold, root+'/thresholds/70.pkl')
            threshold = np.percentile(all_train_distances, 80)
            joblib.dump(threshold, root+'/thresholds/80.pkl')
            threshold = np.percentile(all_train_distances, 90)
            joblib.dump(threshold, root+'/thresholds/90.pkl')
            threshold = np.percentile(all_train_distances, 95)
            joblib.dump(threshold, root+'/thresholds/95.pkl')
            threshold = np.percentile(all_train_distances, 99)
            joblib.dump(threshold, root+'/thresholds/99.pkl')

    print(f"Threshold (95th percentile of kNN distances): {threshold:.4f}")

def predict(dataloader, model, c):
    all_errors = []
    all_labels = []
    tic =time.time()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            z = model(x)  # z: feature embedding
            dist = torch.sum((z - c) ** 2, dim=1)  # Squared L2 distance to center

            all_errors.append(dist.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    print(f"Time taken: {time.time() - tic:.2f} seconds")
    # Stack
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    return all_errors, all_labels

def eval():
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET,ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepSVDD/{dataset}"
        swanlab.init(
            # 设置将记录此次运行的项目信息
            project="OODDetector",
            workspace="sumail",
            # 跟踪超参数和运行元数据
            config={
                "architecture": "DeepSVDD",
                "dataset": attack_type,
                "type": f"{TYPE}"
            }
            )
    
        model = CIFAR10_LeNet().to(DEVICE)
        weights = torch.load(root+"/weights/best_svdd.pth", weights_only=False)
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

        all_labels = []
        all_errors = []
        dataset = ConcatDataset([dataset, attacked_dataset])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        c = init_center_c(dataloader, model, DEVICE)



        # Assume binary labels are available in val_loader or dummy one
        # Evaluate
        # Predict labels
        all_results = []

        all_errors, all_labels = predict(dataloader, model, c)
        eval_bootstrap(all_errors, all_labels, root+"/thresholds")



        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         all_errors, all_labels = predict(dataloader, model, c)

        #         print(f"Using threshold: {thr}")
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
    # get_threshold()

