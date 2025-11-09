from ..shared_parameter import *
DATA_ROOT = COMMON_DATA_ROOT
DATASET = COMMON_DATASET  
DATA_TYPE = "CHAUFFEUR-Track1-Normal"
TYPE = COMMON_TYPE

#ATTACKED_DATA = "/raid/007--Experiments/selforacle/evaluation/pgd_attack/epoch"  # "fgsm" or "pgd"
ATTACK_TYPE = COMMON_ATTACK_TYPE
ATTACK_DATA_ROOT = COMMON_ATTACK_DATA_ROOT
ATTACK_DATA_TYPE = "CHAUFFEUR-Track1-Snow"
ROOT_LOG_DIR = "/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly"

THRESHOLD_ROOT = "/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/Ganomaly/thresholds"
THRESHOLD_RANGE  = COMMON_THRESHOLD  # threshold to be used for evaluation
LATENT_DIM   = 100
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 2e-4
LAMBDA_L1    = 50.0
LAMBDA_ENC   = 1.0   