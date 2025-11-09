
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
from Detectors.Ganomaly.config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import swanlab 
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from Detectors.Ganomaly.ganomaly import Encoder, Decoder, Discriminator
from Detectors.drive_dataset import DrivingOODDataset, DrivingOODDatasetNpy
from Detectors.utils import *
DEVICE = "cuda:0"

root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly/{DATASET}"
def train():
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=transform)

    # Split into train and validation sets (e.g., 80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

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

        # collect per-sample absolute errors for thresholding (TRAIN ONLY)
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

            # per-sample absolute reconstruction error for thresholds
            per_sample_err = torch.mean(torch.abs(real - recon), dim=[1, 2, 3])
            train_errs.extend(per_sample_err.detach().cpu().numpy())

        print(f"[Epoch {epoch}]  G: {np.mean(g_losses):.4f}  D: {np.mean(d_losses):.4f}")

        # ---------- Compute thresholds from TRAIN errors ----------
        train_errs = np.asarray(train_errs)
        mean_train_err = float(train_errs.mean()) if train_errs.size else float("inf")
        thresh_70 = float(np.percentile(train_errs, 70)) if train_errs.size else None
        thresh_80 = float(np.percentile(train_errs, 80)) if train_errs.size else None
        thresh_90 = float(np.percentile(train_errs, 90)) if train_errs.size else None
        thresh_95 = float(np.percentile(train_errs, 95)) if train_errs.size else None
        thresh_99 = float(np.percentile(train_errs, 99)) if train_errs.size else None

        print(f"    ▸ Train mean err: {mean_train_err:.6f} "
              f"| 95-pct: {thresh_95 if thresh_95 is not None else float('nan'):.6f} "
              f"| 99-pct: {thresh_99 if thresh_99 is not None else float('nan'):.6f}")

        # ---------- VALIDATION: compute mean reconstruction error ----------
        enc.eval(); gen.eval(); disc.eval()
        val_errs = []
        with torch.no_grad():
            for vimg, _ in val_loader:
                vimg = vimg.to(DEVICE)
                vz   = enc(vimg)
                vrecon = gen(vz)
                v_per_sample = torch.mean(torch.abs(vimg - vrecon), dim=[1, 2, 3])
                val_errs.extend(v_per_sample.detach().cpu().numpy())

        mean_val_err = float(np.mean(val_errs)) if len(val_errs) else float("inf")
        print(f"    ▸ Val mean err:   {mean_val_err:.6f}")

        # ---------- Checkpoint by BEST VALIDATION ERROR ----------
        if mean_val_err < best_val_err:
            best_val_err = mean_val_err

            # save nets
            torch.save(gen.state_dict(),  root + "/weights/best_generator_ganomaly.pth")
            torch.save(enc.state_dict(),  root + "/weights/best_encoder_ganomaly.pth")
            torch.save(disc.state_dict(), root + "/weights/best_discriminator_ganomaly.pth")

            # save thresholds (still based on TRAIN distribution)
            if thresh_70 is not None:
                joblib.dump(thresh_70, root + '/thresholds/70.pkl')
                joblib.dump(thresh_80, root + '/thresholds/80.pkl')
                joblib.dump(thresh_90, root + '/thresholds/90.pkl')
                joblib.dump(thresh_95, root + '/thresholds/95.pkl')
                joblib.dump(thresh_99, root + '/thresholds/99.pkl')

            print(f"      ✓ New best saved (val mean err {mean_val_err:.6f})")


def predict(dataloader, gen, enc):
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

    return all_scores, all_labels

def eval():
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET,ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly/{dataset}"
        swanlab.init(
            # 设置将记录此次运行的项目信息
            project="OODDetector",
            workspace="sumail",
            # 跟踪超参数和运行元数据
            experiment_name = f"Ganomaly_{TYPE}_{dataset}_{attack_type}",
            config={
                "architecture": "Ganomaly",
                "dataset": attack_type,
                "type": f"{TYPE}"
            }
            )
        enc  = Encoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
        gen  = Decoder(latent_space_dimension=LATENT_DIM).to(DEVICE)
        disc = Discriminator().to(DEVICE)

        enc.load_state_dict(torch.load(root + "/weights/best_encoder_ganomaly.pth",
                                    map_location=DEVICE))
        gen.load_state_dict(torch.load(root + "/weights/best_generator_ganomaly.pth",
                                    map_location=DEVICE))
        enc.eval(); gen.eval()

        transform = transforms.Compose([
            transforms.Resize((160, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        print("Loading dataset...")
        if dataset == "Udacity":
            dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel="", transform=transform)
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

        # Stack

        # Predict labels
        all_results = []
        all_errors, all_labels = predict(dataloader, gen, enc)
        eval_bootstrap(all_errors, all_labels, root+"/thresholds")

        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         all_scores, all_labels = predict(dataloader, gen, enc)
        #         print(f"Using threshold: {thr}")
        #         thr_value = joblib.load(root+f"/thresholds/{str(thr)}.pkl")
        #         pred_labels = (all_scores > thr_value).astype(int)

        #     #     # --- Safe AUC-ROC ---
        #         unique_classes = np.unique(all_labels)
        #         if len(unique_classes) < 2:
        #             auc_roc = np.nan
        #             auc_pr = np.nan
        #         else:
        #             auc_roc = roc_auc_score(all_labels, all_scores)
        #             precision, recall, _ = precision_recall_curve(all_labels, all_scores)
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

