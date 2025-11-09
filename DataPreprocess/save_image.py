import numpy as np
from PIL import Image
import os

def save_npy_as_image(npy_path, output_path, mean=0.5, std=0.5, take_first=True):
    """
    Load an npy image (normalized with Normalize(mean, std) -> e.g. mean=0.5,std=0.5)
    and save to output_path as PNG.

    Accepts shapes:
      - (N, C, H, W) or (N, H, W, C)  -> by default saves the first image (set take_first=False to save all)
      - (C, H, W)
      - (H, W, C)

    mean/std can be scalar or sequence; this function assumes same mean/std for all channels
    when a scalar is provided (typical for [0.5,0.5,0.5]).
    """
    # load
    arr = np.load(npy_path, allow_pickle=False)
    print(f"Loaded {npy_path} with shape {arr.shape} and dtype {arr.dtype}")
    print("shape:", arr.shape)
    print("min, max:", arr.min(), arr.max())

    # normalize mean/std as lists per channel
    # If user gives scalars, expand later based on channels
    # We'll only need mean/std for denormalization: x_denorm = x * std + mean
    # Here we accept scalar mean/std for convenience (most common case).
    # Later we will apply per-channel if needed.
    if isinstance(mean, (int, float)):
        mean_vals = None  # fill after we know channel count
    else:
        mean_vals = list(mean)
    if isinstance(std, (int, float)):
        std_vals = None
    else:
        std_vals = list(std)

    # Convert to a list of HWC images (values in normalized space)
    hwc_list = []

    if arr.ndim == 4:
        # Could be NCHW or NHWC. Detect which by checking channel dimension size (1 or 3).
        if arr.shape[1] in (1, 3):
            # N, C, H, W -> convert each to H, W, C
            for i in range(arr.shape[0]):
                hwc_list.append(np.transpose(arr[i], (1, 2, 0)))
        elif arr.shape[3] in (1, 3):
            # N, H, W, C already
            for i in range(arr.shape[0]):
                hwc_list.append(arr[i])
        else:
            raise ValueError(f"Unrecognized 4D array shape {arr.shape}. Expected channels=1 or 3 in axis 1 or 3.")
    elif arr.ndim == 3:
        # Could be (C,H,W) or (H,W,C)
        if arr.shape[0] in (1, 3):
            # C,H,W -> H,W,C
            hwc_list.append(np.transpose(arr, (1, 2, 0)))
        elif arr.shape[2] in (1, 3):
            # H,W,C already
            hwc_list.append(arr)
        else:
            raise ValueError(f"Unrecognized 3D array shape {arr.shape}. Expected channel dim to be 1 or 3.")
    else:
        raise ValueError(f"Unsupported array with ndim={arr.ndim}. Expected 3 or 4 dims.")

    # decide mean/std per channel now
    c = hwc_list[0].shape[2]
    if mean_vals is None:
        mean_vals = [mean] * c
    if std_vals is None:
        std_vals = [std] * c
    if len(mean_vals) != c or len(std_vals) != c:
        raise ValueError(f"Mean/std length ({len(mean_vals)}/{len(std_vals)}) doesn't match channels ({c})")

    # helper to denormalize a single HWC image
    def denormalize_hwc(img_hwc):
        img = img_hwc.astype(np.float32)
        for ch in range(c):
            img[..., ch] = img[..., ch] * std_vals[ch] + mean_vals[ch]
        # Clip to [0,1]
        img = np.clip(img, 0.0, 1.0)
        return img

    # Save either first only or all
    out_paths = []
    if take_first:
        out_list = [(hwc_list[0], output_path)]
    else:
        # if output_path ends with an extension, remove it and append index
        base, ext = os.path.splitext(output_path)
        out_list = []
        for idx, img_hwc in enumerate(hwc_list):
            out_list.append((img_hwc, f"{base}_{idx}{ext or '.png'}"))

    for img_hwc, out_path in out_list:
        img01 = denormalize_hwc(img_hwc)     # HWC in [0,1]
        img_u8 = (img01 * 255.0).round().astype(np.uint8)
        # If single-channel, PIL expects HxW
        if img_u8.shape[2] == 1:
            pil = Image.fromarray(img_u8[:, :, 0], mode="L")
        else:
            pil = Image.fromarray(img_u8, mode="RGB")
        pil.save(out_path)
        print(f"Saved: {out_path} (shape {img_u8.shape}, dtype {img_u8.dtype})")
        out_paths.append(out_path)

    return out_paths

# ---- Example usage ----
npy_file = "/raid/007--Experiments/selforacle/drive_dataset/test/fgsm_attack_dave2v1/adv_000_01.npy"
out_files = save_npy_as_image(npy_file, "output.png")
