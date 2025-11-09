# fanogan_pipeline.py
# Single-file f-AnoGAN pipeline compatible with your current training/eval script.
# Keeps your datasets, loaders, Swanlab logging, thresholding loop, and CLI switches.
# Structure follows GitHub f-AnoGAN: Stage-I WGAN-GP (G,D), Stage-II Encoder with feature-matching,
# anomaly score = residual + discriminator feature distance. (Refs: A03ki/tSchlegl) [see chat for citations]

import os, time, joblib, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
from Detectors.FAnoGAN.config import *
# ==== Your deps (unchanged) ====
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from Detectors.udacity_dataset import UdacityImageDataset, UdacityImageTestDataset, UdacityImageAttackDataset, AnomalImageDataset
from Detectors.drive_dataset import DrivingOODDataset, DrivingOODDatasetNpy
import swanlab
from Detectors.utils import *
# ==== Config (edit or import from your Ganomaly/config if you prefer) ====
# If you already have Detectors/Ganomaly/config.py with these names, you can import from there.
# Here we define reasonable defaults and let env override.
DEVICE              = os.getenv("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu"

# Training epochs (you can keep same as your GANomaly runs)
EPOCHS_STAGE_GAN    = int(os.getenv("EPOCHS_STAGE_GAN", "30"))
EPOCHS_STAGE_ENC    = int(os.getenv("EPOCHS_STAGE_ENC", "30"))
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS         = int(os.getenv("NUM_WORKERS", "4"))

# f-AnoGAN hyperparams (faithful to repos/paper; adjust as needed)
Z_DIM       = int(os.getenv("Z_DIM", "100"))
LR_G        = float(os.getenv("LR_G", "1e-4"))
LR_D        = float(os.getenv("LR_D", "1e-4"))
LR_E        = float(os.getenv("LR_E", "1e-4"))
BETA1       = float(os.getenv("BETA1", "0.5"))
BETA2       = float(os.getenv("BETA2", "0.9"))
N_CRITIC    = int(os.getenv("N_CRITIC", "5"))
GP_LAMBDA   = float(os.getenv("GP_LAMBDA", "10.0"))
LAMBDA_RES  = float(os.getenv("LAMBDA_RES", "1.0"))   # residual loss weight
LAMBDA_FM   = float(os.getenv("LAMBDA_FM", "0.1"))    # feature-matching weight

# Paths compatible with your previous 'root'
root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/FAnoGAN/{DATASET}"
# os.makedirs(os.path.join(root, "weights"), exist_ok=True)
# os.makedirs(os.path.join(root, "thresholds"), exist_ok=True)

# ==== Models: DCGAN-style G/D with WGAN-GP discriminator head; Encoder with conv downsampling ====
class Generator(nn.Module):
    """DCGAN-style deconv stack producing (3, 160, 320) exactly."""
    def __init__(self, latent_dim=Z_DIM, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # 1) Seed: 1x1 -> 5x10
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=(5, 10), stride=1, padding=0, bias=False),  # 5x10
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),

            # 2) 5x10 -> 10x20
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 10x20
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            # 3) 10x20 -> 20x40
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 20x40
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            # 4) 20x40 -> 40x80
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),  # 40x80
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            # 5) 40x80 -> 80x160
            nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),  # 80x160
            nn.BatchNorm2d(ngf // 2), nn.ReLU(True),

            # 6) 80x160 -> 160x320
            nn.ConvTranspose2d(ngf // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),  # 160x320
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """WGAN-GP critic with scalar output and feature vector for f-AnoGAN."""
    def __init__(self, ndf=64):
        super().__init__()
        def block(in_c, out_c, k, s, p, instnorm=True):
            layers = [nn.Conv2d(in_c, out_c, k, s, p)]
            if instnorm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, ndf,   4, 2, 1, instnorm=False),  # 80x160
            block(ndf, ndf*2, 4, 2, 1),                # 40x80
            block(ndf*2, ndf*4, 4, 2, 1),              # 20x40
            block(ndf*4, ndf*8, 4, 2, 1),              # 10x20
            block(ndf*8, ndf*8, 4, 2, 1),              # 5x10
        )
        # Resolution-agnostic head: GAP -> Linear -> scalar
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.adv_fc = nn.Linear(ndf * 8, 1)

    def forward(self, x, return_features=False):
        h = self.features(x)                  # (B, ndf*8, H, W)
        gap = self.gap(h).view(x.size(0), -1) # (B, ndf*8)
        out = self.adv_fc(gap).squeeze(1)     # (B,)
        if return_features:
            # Use GAP'ed vector as the feature for feature matching (common in f-AnoGAN)
            f = gap                           # (B, ndf*8)
            return out, f
        return out


class Encoder(nn.Module):
    """Image -> z encoder trained in Stage-II via residual + feature-matching loss."""
    def __init__(self, latent_dim=Z_DIM, nef=64):
        super().__init__()
        def block(in_c, out_c, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            block(3,   nef,   4, 2, 1),   # 80x160
            block(nef, nef*2, 4, 2, 1),   # 40x80
            block(nef*2, nef*4, 4, 2, 1), # 20x40
            block(nef*4, nef*8, 4, 2, 1), # 10x20
            block(nef*8, nef*8, 4, 2, 1), # 5x10
        )
        self.fc_mu = nn.Linear(nef*8, latent_dim)
    def forward(self, x):
        h = self.net(x)                   # (B, nef*8, 5, 10)
        h = torch.mean(h, dim=(2,3))      # GAP -> (B, nef*8)
        z = self.fc_mu(h)
        return z

# ==== Utils ====
def _grad_penalty(disc, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    inter = real * alpha + fake * (1 - alpha)
    inter.requires_grad_(True)
    d_inter = disc(inter)
    grads = torch.autograd.grad(
        outputs=d_inter, inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ==== Data transforms (unchanged from your script) ====
def build_transform():
    return transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# ==== Stage-I & Stage-II Training ====
def train():
    transform = build_transform()

    # Load full dataset (ID only)
    if DATASET == "Udacity":
        base_dir = '/raid/007--Experiments/selforacle/training_data'
        dataset = UdacityImageDataset(base_dir=base_dir, transform=transform)
    else:
        dataset = DrivingOODDataset(base_dir=DATA_ROOT, label=0, transform=transform)

    # Split into train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ----- Stage-I: train WGAN-GP on normals -----
    G = Generator(latent_dim=Z_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    best_val_proxy = float("inf")
    for epoch in range(1, EPOCHS_STAGE_GAN + 1):
        G.train(); D.train()
        d_losses, g_losses = [], []
        for x, _ in tqdm(train_loader, desc=f"Stage-I (GAN) E{epoch}/{EPOCHS_STAGE_GAN}"):
            x = x.to(DEVICE)
            b = x.size(0)

            # Train critic N_CRITIC steps
            for _ in range(N_CRITIC):
                z = torch.randn(b, Z_DIM, device=DEVICE)
                fake = G(z).detach()
                d_real = D(x)
                d_fake = D(fake)
                gp = _grad_penalty(D, x, fake, DEVICE) * GP_LAMBDA
                d_loss = (d_fake - d_real).mean() + gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                d_losses.append(d_loss.item())

            # Train generator
            z = torch.randn(b, Z_DIM, device=DEVICE)
            fake = G(z)
            g_loss = -D(fake).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            g_losses.append(g_loss.item())

        # quick proxy val (not used for scoring)
        G.eval(); D.eval()
        with torch.no_grad():
            v_losses = []
            for vimg, _ in val_loader:
                vimg = vimg.to(DEVICE)
                vz   = torch.randn(vimg.size(0), Z_DIM, device=DEVICE)
                vrecon = G(vz)
                v_losses.append(torch.mean(torch.abs(vrecon - vimg)).item())
        val_proxy = float(np.mean(v_losses)) if v_losses else float("inf")
        print(f"[Stage-I] Epoch {epoch}  D:{np.mean(d_losses):.4f}  G:{np.mean(g_losses):.4f}  proxy_rec:{val_proxy:.4f}")

        if val_proxy < best_val_proxy:
            best_val_proxy = val_proxy
            torch.save(G.state_dict(),  os.path.join(root, "weights/best_generator_fanogan.pth"))
            torch.save(D.state_dict(),  os.path.join(root, "weights/best_discriminator_fanogan.pth"))
            print(f"  ✓ Saved Stage-I best (proxy_rec {val_proxy:.4f})")

    # ----- Stage-II: train Encoder with residual + feature-matching -----
    E = Encoder(latent_dim=Z_DIM).to(DEVICE)
    for p in G.parameters(): p.requires_grad_(False)
    for p in D.parameters(): p.requires_grad_(False)
    opt_E = optim.Adam(E.parameters(), lr=LR_E, betas=(BETA1, BETA2))

    best_val_res = float("inf")
    for epoch in range(1, EPOCHS_STAGE_ENC + 1):
        E.train()
        train_residuals = []
        e_losses = []
        for x, _ in tqdm(train_loader, desc=f"Stage-II (Encoder) E{epoch}/{EPOCHS_STAGE_ENC}"):
            x = x.to(DEVICE)
            z = E(x)
            x_hat = G(z)
            # residual (per-sample)
            res = torch.mean(torch.abs(x - x_hat), dim=(1,2,3))
            # feature-matching (per-sample)
            _, f_real = D(x, return_features=True)
            _, f_fake = D(x_hat, return_features=True)
            fm = torch.mean((f_real - f_fake)**2, dim=1)

            loss = (LAMBDA_RES * res + LAMBDA_FM * fm).mean()
            opt_E.zero_grad(); loss.backward(); opt_E.step()
            e_losses.append(loss.item())
            train_residuals.extend(res.detach().cpu().numpy())

        # thresholds from TRAIN residuals (backward compatible with your eval loop)
        train_residuals = np.asarray(train_residuals)
        thresh = {p: float(np.percentile(train_residuals, p)) for p in THRESHOLD_RANGE} if train_residuals.size else {}

        # simple validation: mean residual
        E.eval()
        val_residuals = []
        with torch.no_grad():
            for vimg, _ in val_loader:
                vimg = vimg.to(DEVICE)
                vz = E(vimg); vrecon = G(vz)
                v_res = torch.mean(torch.abs(vimg - vrecon), dim=(1,2,3))
                val_residuals.extend(v_res.detach().cpu().numpy())
        mean_val_res = float(np.mean(val_residuals)) if val_residuals else float("inf")
        print(f"[Stage-II] Epoch {epoch}  E_loss:{np.mean(e_losses):.4f}  ValResidual:{mean_val_res:.6f}")

        if mean_val_res < best_val_res:
            best_val_res = mean_val_res
            torch.save(G.state_dict(),  os.path.join(root, "weights/best_generator_fanogan.pth"))
            torch.save(E.state_dict(),  os.path.join(root, "weights/best_encoder_fanogan.pth"))
            torch.save(D.state_dict(),  os.path.join(root, "weights/best_discriminator_fanogan.pth"))
            for k, v in thresh.items():
                joblib.dump(v, os.path.join(root, f"thresholds/{k}.pkl"))
            print(f"  ✓ Saved Stage-II best (val mean residual {mean_val_res:.6f}) and thresholds")


# ==== Inference utilities ====
@torch.no_grad()
def predict(dataloader, G, E, D):
    tic = time.time()
    all_scores_res, all_scores_comb, all_labels = [], [], []
    for x, y in tqdm(dataloader, desc="Testing"):
        x = x.to(DEVICE)
        z = E(x)
        x_hat = G(z)
        # residual
        res = torch.mean(torch.abs(x - x_hat), dim=(1,2,3))
        # feature distance
        _, f_real = D(x, return_features=True)
        _, f_fake = D(x_hat, return_features=True)
        fm = torch.mean((f_real - f_fake)**2, dim=1)
        # combined score (for AUC ranking)
        comb = LAMBDA_RES * res + LAMBDA_FM * fm

        all_scores_res.extend(res.detach().cpu().numpy())
        all_scores_comb.extend(comb.detach().cpu().numpy())
        all_labels.extend(y.numpy())
    print(f"Time taken: {time.time() - tic:.2f} seconds")
    return np.asarray(all_scores_res), np.asarray(all_scores_comb), np.asarray(all_labels)


def eval():
    for attack_type, dataset, attack_data_root in zip(ATTACK_TYPE, DATASET,ATTACK_DATA_ROOT):
        root = f"/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/FAnoGAN/{dataset}"
        swanlab.init(
        # 设置将记录此次运行的项目信息
        project="OODDetector",
        workspace="sumail",
        # 跟踪超参数和运行元数据
        config={
            "architecture": "FAnoGAN",
            "dataset": attack_data_root,
            "type": f"{TYPE}"
        }
        )
        print("Iteration: ", attack_type, dataset, attack_data_root)

    # load models
        E = Encoder(latent_dim=Z_DIM).to(DEVICE)
        G = Generator(latent_dim=Z_DIM).to(DEVICE)
        D = Discriminator().to(DEVICE)
        E.load_state_dict(torch.load(os.path.join(root, "weights/best_encoder_fanogan.pth"),      map_location=DEVICE))
        G.load_state_dict(torch.load(os.path.join(root, "weights/best_generator_fanogan.pth"),    map_location=DEVICE))
        D.load_state_dict(torch.load(os.path.join(root, "weights/best_discriminator_fanogan.pth"), map_location=DEVICE))
        E.eval(); G.eval(); D.eval()

        transform = build_transform()

        # datasets exactly like your code
        print("Loading dataset...")
        if dataset == "Udacity":
            dataset = UdacityImageTestDataset(base_dir=DATA_ROOT, data=[DATA_TYPE], mode="clean", fmodel="", transform=transform)
            if attack_type == "weather":
                rt, weather_type = attack_data_root
                print(rt, weather_type)
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
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        all_results = []
        scores_res, scores_comb, labels = predict(dataloader, G, E, D)
        eval_bootstrap(scores_res, labels, root+"/thresholds", commb=scores_comb)


        # for thr_range in COMMON_THRESHOLD:
        #     value = thr_range[0]
        #     for thr in thr_range:
        #         scores_res, scores_comb, labels = predict(dataloader, G, E, D)
        #         print(f"Using residual threshold key: {thr}")
        #         thr_value = joblib.load(os.path.join(root, f"thresholds/{str(thr)}.pkl"))

        #         # Binary decision on residual (compatible with saved thresholds)
        #         pred_labels = (scores_res > thr_value).astype(int)

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



if __name__ == "__main__":
    # Choose behavior via TYPE
    if TYPE == "eval":
        eval()
    else:
        train()
