from ..shared_parameter import *

# DATA_ROOT = COMMON_DATA_ROOT
DATA_ROOT = COMMON_DATA_ROOT
DATA_TYPE = "CHAUFFEUR-Track1-Normal"
DATASET = COMMON_DATASET  
TYPE = COMMON_TYPE

#ATTACKED_DATA = "/raid/007--Experiments/selforacle/evaluation/pgd_attack/epoch"  # "fgsm" or "pgd"
ATTACK_TYPE = COMMON_ATTACK_TYPE
ATTACK_DATA_ROOT = COMMON_ATTACK_DATA_ROOT
ATTACK_DATA_TYPE = "CHAUFFEUR-Track1-Snow"
THRESHOLD_ROOT = "/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/Detectors/DeepRoad/thresholds"
THRESHOLD_RANGE  = COMMON_THRESHOLD  # threshold to be used for evaluation