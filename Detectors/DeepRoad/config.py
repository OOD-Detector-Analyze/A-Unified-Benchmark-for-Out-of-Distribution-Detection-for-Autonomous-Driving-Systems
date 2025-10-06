DATA_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/evaluation_data"
DATA_TYPE = "CHAUFFEUR-Track1-Normal"
DATASET = "Udacity"  
TYPE = "eval"

#ATTACKED_DATA = "/raid/007-Xiangyu-Experiments/selforacle/evaluation/pgd_attack/epoch"  # "fgsm" or "pgd"
ATTACK_TYPE = "attack"
ATTACK_DATA_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/evaluation_data"
ATTACK_DATA_TYPE = "CHAUFFEUR-Track1-Snow"
ATTACKED_DATA = "/raid/007-Xiangyu-Experiments/selforacle/evaluation/spsa_attack/epoch"
THRESHOLD_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/thresholds"
THRESHOLD_RANGE  = [70,80,90,95,99]  # threshold to be used for evaluation
PCA_FILE  = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/weights/pca_model.pkl"
TRAIN_NPY = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/weights/train_vecs.npy"