import logging
from scipy.stats import gamma

import cv2
import matplotlib.image as mpimg
import numpy as np
import torch
from numpy.random import default_rng
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
logger = logging.Logger("utils")

# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 33, 100, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 80, 160, 3

ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_CHANNELS = 160, 320, 3

# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image_dir = data_dir
    local_path = "/".join(image_file.split("/")[-4:-1]) + "/" + image_file.split("/")[-1]
    img_path = "{0}/{1}".format(image_dir, local_path)
    return mpimg.imread(img_path)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    # choice = np.random.choice(3)
    # if choice == 0:
    #     return load_image(data_dir, left), steering_angle + 0.2
    # elif choice == 1:
    #     return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image


def random_translate(image, range_x, range_y):
    """
    Randomly shift the image virtually and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    # xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    xm, ym = np.mgrid[0:ORIGINAL_IMAGE_HEIGHT, 0:ORIGINAL_IMAGE_WIDTH] # fix

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(picture, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # image = load_image(data_dir, picture)
    image = random_flip(picture)
    image = random_translate(image, range_x=range_x, range_y=range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image

def load_img_from_path(data_dir, image_name, is_gray_scale):
    """
    Loads an image from the provided path and resizes it to the size specified in the utils class
    :param image_name:  path to image within dataset
    :param args: run args (required for args.data_dir and grayscale settings)
    :return:
    """
    img = load_image(data_dir, image_name)
    if is_gray_scale:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return __normalize_and_reshape(img, is_gray_scale)


def __normalize_and_reshape(x, is_gray_scale):
    img = cv2.resize(x, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    if not is_gray_scale:
        img = img.astype('float32') / 255.
    img = img.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    return img


# def attack(att_model, attacks, image, label, acceptable_perturbation_rate=0.01, verbose=False):
#         # attacks = []
#         # if attack_name == 'sp':
#         #     attacks = [
#         #         fb.attacks.SaltAndPepperNoiseAttack()
#         #     ]
#         # elif attack_name == 'ddn':
#         #     attacks = [
#         #         fb.attacks.DDNAttack()
#         #     ]
#         # elif attack_name == 'dp':
#         #     attacks = [
#         #         fb.attacks.LinfDeepFoolAttack(steps=100)
#         #     ]
#         # elif attack_name == 'l2':
#         #     attacks = [
#         #         fb.attacks.L2CarliniWagnerAttack(steps=100)
#         #     ]
#         # elif attack_name == 'sp_ddn':
#         #     attacks = [
#         #         fb.attacks.SaltAndPepperNoiseAttack(),
#         #         fb.attacks.DDNAttack()
#         #     ]
#         # else:
#         #     log_info(logger, 'Invalid attack type.')

#         image_, restore_type = ep.astensor_(image)
#         acceptable_perturbation = restore_type(
#             ep.norms.lp(flatten(image_), 2, axis=-1)) * acceptable_perturbation_rate
#         criterion = fb.criteria.TargetedMisclassification(label)
#         perturbations = {}
#         adv_images = {}
#         output = []
#         ## alter images using the attacks in order, try second attack only if the first attack is failed
#         for i in range(len(attacks)):
#             adv_images[i], _, is_adv = attacks[i](att_model, image, criterion, epsilons=None)
#             perturbations[i] = fb.distances.l2(image, adv_images[i])
#             lowest_perturbation_idx = min(perturbations, default=-1, key=perturbations.get)
#             return adv_images[lowest_perturbation_idx]

# def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
#     return x.flatten(start=keep)


def calc_and_store_thresholds(losses: np.array, model_class: str) -> dict:
    """
    Calculates all thresholds stores them on a file system
    :param losses: array of shape (n,),
                    where n is the number of training data points, containing the losses calculated for these points
    :param model_class: the identifier of the anomaly detector type
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """

    print("Fitting reconstruction error distribution of %s using Gamma distribution params" % model_class)

    shape, loc, scale = gamma.fit(losses, floc=0)

    thresholds = {}

    #conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    conf_intervals = [0.7, 0.8, 0.9, 0.95, 0.99]

    print("Creating thresholds using the confidence intervals: %s" % conf_intervals)

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    print(thresholds)
    return thresholds


def bootstrap_metric(y_true, y_score, thr, n_boot=1000, seed=42):
    rng = default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    auc_roc_list, auc_pr_list, f1_list = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)  # bootstrap sample
        yt = y_true[idx]
        ys = y_score[idx]

        # AUC-ROC
        if len(np.unique(yt)) > 1:
            auc_roc_list.append(roc_auc_score(yt, ys))
            precision, recall, _ = precision_recall_curve(yt, ys)
            auc_pr_list.append(auc(recall, precision))
        else:
            auc_roc_list.append(np.nan)
            auc_pr_list.append(np.nan)

        # F1 at given threshold
        pred = (ys > thr).astype(int)
        f1_list.append(f1_score(yt, pred, zero_division=0))

    # Convert to arrays
    auc_roc_list = np.array(auc_roc_list, dtype=float)
    auc_pr_list  = np.array(auc_pr_list, dtype=float)
    f1_list      = np.array(f1_list, dtype=float)

    def ci(a):
        a = a[~np.isnan(a)]
        if len(a) == 0: return (np.nan, np.nan, np.nan)
        return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

    return {
        "AUC-ROC": ci(auc_roc_list),
        "AUC-PR":  ci(auc_pr_list),
        "F1":      ci(f1_list)
    }




def eval_bootstrap(all_errors, all_labels, threshold_dir, percentiles=[70,80,90,95,99], n_boot=1000, commb=None):
    results = []

    for p in percentiles:
        thr_value = float(joblib.load(f"{threshold_dir}/{p}.pkl"))
        pred = (all_errors > thr_value).astype(int)
        if commb is None:
            if len(np.unique(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, all_errors)
                precision, recall, _ = precision_recall_curve(all_labels, all_errors)
                auc_pr = auc(recall, precision)
            else:
                auc_roc = auc_pr = np.nan

            f1 = f1_score(all_labels, pred, zero_division=0)

            boot = bootstrap_metric(all_labels, all_errors, thr_value, n_boot=n_boot)

            results.append({
                "threshold": p,
                "thr_value": thr_value,
                "AUC-ROC": auc_roc,
                "AUC-PR": auc_pr,
                "F1": f1,
                "AUC-ROC_boot_mean": boot["AUC-ROC"][0],
                "AUC-ROC_CI": boot["AUC-ROC"][1:],
                "AUC-PR_boot_mean": boot["AUC-PR"][0],
                "AUC-PR_CI": boot["AUC-PR"][1:],
                "F1_boot_mean": boot["F1"][0],
                "F1_CI": boot["F1"][1:]
            })
        else:
            if len(np.unique(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, commb)
                precision, recall, _ = precision_recall_curve(all_labels, commb)
                auc_pr = auc(recall, precision)
            else:
                auc_roc = auc_pr = np.nan

            f1 = f1_score(all_labels, pred, zero_division=0)

            boot = bootstrap_metric(all_labels, commb, thr_value, n_boot=n_boot)

            results.append({
                "threshold": p,
                "thr_value": thr_value,
                "AUC-ROC": auc_roc,
                "AUC-PR": auc_pr,
                "F1": f1,
                "AUC-ROC_boot_mean": boot["AUC-ROC"][0],
                "AUC-ROC_CI": boot["AUC-ROC"][1:],
                "AUC-PR_boot_mean": boot["AUC-PR"][0],
                "AUC-PR_CI": boot["AUC-PR"][1:],
                "F1_boot_mean": boot["F1"][0],
                "F1_CI": boot["F1"][1:]
            })


    format_print(results)

    return results


def format_print(result):
    normal_res = {"AUC-R":0, "AUC-P":0, "F1":[]}
    bootstrap_res = {"AUC-ROC_boot_mean":0,"AUC-ROC_CI":0, "AUC-PR_boot_mean":0, "AUC-PR_CI":0, "F1_boot_mean":[], "F1_CI":[]}
    for idx, ele in enumerate(result):
        normal_res["AUC-R"]=ele["AUC-ROC"] 
        normal_res["AUC-P"]=ele["AUC-PR"]
        normal_res["F1"].append(ele["F1"])

        bootstrap_res["AUC-ROC_boot_mean"]=ele["AUC-ROC_boot_mean"] 
        bootstrap_res["AUC-ROC_CI"]=ele["AUC-ROC_CI"]
        bootstrap_res["AUC-PR_boot_mean"]=ele["AUC-PR_boot_mean"]
        bootstrap_res["AUC-PR_CI"]=ele["AUC-PR_CI"]
        bootstrap_res["F1_boot_mean"].append(ele["F1_boot_mean"])
        bootstrap_res["F1_CI"].append(ele["F1_CI"])
        
    print("---------------------------------FINAL RESULT PRINT---------------------------------------------------------")
    print(
        f"Normal outcome: {float(normal_res['AUC-R']):.2f} & "
        f"{float(normal_res['AUC-P']):.2f} & "
        f"{float(normal_res['F1'][0]):.2f} & "
        f"{float(normal_res['F1'][1]):.2f} & "
        f"{float(normal_res['F1'][2]):.2f} & "
        f"{float(normal_res['F1'][3]):.2f} & "
        f"{float(normal_res['F1'][4]):.2f} //"
    )

    print(
        f"Bootstrap outcome: {float(bootstrap_res['AUC-ROC_boot_mean']):.2f} & "
        f"{(bootstrap_res['AUC-ROC_CI'])} & "
        f"{float(bootstrap_res['AUC-PR_boot_mean']):.2f} & "
        f"{(bootstrap_res['AUC-PR_CI'])} & "
        f"{float(bootstrap_res['F1_boot_mean'][0]):.2f} & "
        f"{(bootstrap_res['F1_CI'][0])} & "
        f"{float(bootstrap_res['F1_boot_mean'][1]):.2f} & "
        f"{(bootstrap_res['F1_CI'][1])} & "
        f"{float(bootstrap_res['F1_boot_mean'][2]):.2f} & "
        f"{(bootstrap_res['F1_CI'][2])} & "
        f"{float(bootstrap_res['F1_boot_mean'][3]):.2f} & "
        f"{(bootstrap_res['F1_CI'][3])} & "
        f"{float(bootstrap_res['F1_boot_mean'][4]):.2f} & "
        f"{(bootstrap_res['F1_CI'][4])} & "
    )
    