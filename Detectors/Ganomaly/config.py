DATA_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/evaluation_data"
DATA_TYPE = "CHAUFFEUR-Track1-Normal"
DATASET = "Udacity"  
TYPE = "eval"

#ATTACKED_DATA = "/raid/007-Xiangyu-Experiments/selforacle/evaluation/pgd_attack/epoch"  # "fgsm" or "pgd"
ATTACK_TYPE = "attack"
ATTACK_DATA_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/evaluation_data"
ATTACK_DATA_TYPE = "CHAUFFEUR-Track1-Snow"
ATTACKED_DATA = "/raid/007-Xiangyu-Experiments/selforacle/evaluation/spsa_attack/epoch"
ROOT_LOG_DIR = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly"

THRESHOLD_ROOT = "/raid/007-Xiangyu-Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly/thresholds"
THRESHOLD_RANGE  = [70,80,90,95,99]  # threshold to be used for evaluation
LATENT_DIM   = 100
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 2e-4
LAMBDA_L1    = 50.0
LAMBDA_ENC   = 1.0   