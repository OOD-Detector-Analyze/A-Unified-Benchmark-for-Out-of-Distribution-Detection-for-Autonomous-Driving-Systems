
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
from Detectors.Ganomaly.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from Detectors.Ganomaly.ganomaly import Encoder, Decoder, Discriminator
DEVICE = "cuda:0"
def train():
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
 
    enc  = Encoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
    gen  = Decoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    enc_opt  = optim.Adam(enc.parameters(),  lr=LR, betas=(0.5, 0.999))
    gen_opt  = optim.Adam(gen.parameters(),  lr=LR, betas=(0.5, 0.999))
    disc_opt = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    mse = nn.MSELoss()

    best_val_err = float("inf")

    for epoch in range(1, EPOCHS + 1):
        enc.train(); gen.train(); disc.train()
        g_losses, d_losses = [], []

        # ①   用来统计本 epoch 的训练误差
        train_errs = []

        for real, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            real = real.to(DEVICE)
            bs   = real.size(0)
            valid = torch.ones(bs, 1, device=DEVICE)
            fake  = torch.zeros(bs, 1, device=DEVICE)

            # ---------- Generator + Encoder ----------
            z      = enc(real)
            recon  = gen(z)
            d_fake, _ = disc(recon)

            adv_loss = bce(d_fake, valid)
            l1_loss  = l1(recon, real)
            e_real   = z
            e_recon  = enc(recon)
            enc_loss = mse(e_real, e_recon)

            g_total  = adv_loss + LAMBDA_L1 * l1_loss + LAMBDA_ENC * enc_loss

            gen_opt.zero_grad(); enc_opt.zero_grad()
            g_total.backward()
            gen_opt.step(); enc_opt.step()

            # ---------- Discriminator ----------
            d_real, _ = disc(real.detach())
            d_fake, _ = disc(recon.detach())
            d_loss = 0.5 * (bce(d_real, valid) + bce(d_fake, fake))

            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()

            g_losses.append(g_total.item())
            d_losses.append(d_loss.item())

            # ①-continued：记录每个样本的绝对误差
            per_sample_err = torch.mean(torch.abs(real - recon), dim=[1,2,3])
            train_errs.extend(per_sample_err.detach().cpu().numpy())

        print(f"[Epoch {epoch}]  G: {np.mean(g_losses):.4f}  D: {np.mean(d_losses):.4f}")

        # ②  —— 用训练误差算阈值
        train_errs = np.asarray(train_errs)
        mean_train_err = train_errs.mean()
        thresh_70 = float(np.percentile(train_errs, 70))
        thresh_80 = float(np.percentile(train_errs, 80))
        thresh_90 = float(np.percentile(train_errs, 90))
        thresh_95 = float(np.percentile(train_errs, 95))
        thresh_99 = float(np.percentile(train_errs, 99))

        print(f"    ▸ Train mean err: {mean_train_err:.6f} "
              f"|  95-pct: {thresh_95:.6f}  99-pct: {thresh_99:.6f}")

        # ③+④  —— Checkpoint + 一次性保存多阈值
        if mean_train_err < best_val_err:
            best_val_err = mean_train_err
            torch.save(gen.state_dict(),  ROOT_LOG_DIR + "/weights/best_generator_ganomaly.pth")
            torch.save(enc.state_dict(),  ROOT_LOG_DIR + "/weights/best_encoder_ganomaly.pth")
            torch.save(disc.state_dict(), ROOT_LOG_DIR + "/weights/best_discriminator_ganomaly.pth")
            joblib.dump(thresh_70, ROOT_LOG_DIR+'/thresholds/70.pkl')
            joblib.dump(thresh_80, ROOT_LOG_DIR+'/thresholds/80.pkl')
            joblib.dump(thresh_90, ROOT_LOG_DIR+'/thresholds/90.pkl')
            joblib.dump(thresh_95, ROOT_LOG_DIR+'/thresholds/95.pkl')
            joblib.dump(thresh_99, ROOT_LOG_DIR+'/thresholds/99.pkl')
            print(f"      ✓ New best saved (mean err {mean_train_err:.6f})")


def eval():
    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        experiment_name = f"Ganomaly_{TYPE}_{DATASET}",
        config={
            "architecture": "Ganomaly",
            "dataset": DATASET,
            "type": TYPE
        }
        )
    enc  = Encoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
    gen  = Decoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
    disc = Discriminator().to(DEVICE)

    enc.load_state_dict(torch.load(ROOT_LOG_DIR + "/weights/best_encoder_ganomaly.pth",
                                   map_location=DEVICE))
    gen.load_state_dict(torch.load(ROOT_LOG_DIR + "/weights/best_generator_ganomaly.pth",
                                   map_location=DEVICE))
    enc.eval(); gen.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
   
   
    print("Loading dataset...")
    print("Data root:", DATA_ROOT)
    print("Data type:", DATA_TYPE)
    print("Attacked data:", ATTACK_TYPE)
    dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel="", transform=transform)
    if ATTACK_TYPE == "weather":
        attacked_dataset = UdacityImageAttackDataset(base_dir=ATTACK_DATA_ROOT, data=[ATTACK_DATA_TYPE], transform=transform)

    elif ATTACK_TYPE == "attack":
        attacked_dataset = AnomalImageDataset(ATTACKED_DATA, transform=transform)
    dataset = ConcatDataset([dataset, attacked_dataset])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    tic =time.time()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Testing"):
            x = x.to(DEVICE)
            recon = gen(enc(x))
            errors = torch.mean(torch.abs(x - recon), dim=[1, 2, 3])
            all_scores.extend(errors.cpu().numpy())
            all_labels.extend(y.numpy())

    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)
    print(f"Time taken: {time.time() - tic:.2f} seconds")
    # Stack

    # Predict labels
    all_results = []

    for thr in THRESHOLD_RANGE:
        print(f"Using threshold: {thr}")
        thr_value = joblib.load(f"{THRESHOLD_ROOT}/{str(thr)}.pkl")
        pred_labels = (all_scores > thr_value).astype(int)

    #     # --- Safe AUC-ROC ---
        unique_classes = np.unique(all_labels)
        if len(unique_classes) < 2:
            auc_roc = np.nan
            auc_pr = np.nan
        else:
            auc_roc = roc_auc_score(all_labels, all_scores)
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
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

