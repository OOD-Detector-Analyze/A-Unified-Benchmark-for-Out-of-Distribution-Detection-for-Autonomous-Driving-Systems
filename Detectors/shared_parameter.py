COMMON_DATASET = ["Drive","Drive","Drive","Drive","Drive","Drive","Drive","Drive","Drive"]
# COMMON_DATASET = ["Udacity","Udacity","Udacity","Udacity","Udacity","Udacity","Udacity","Udacity","Udacity"]
#COMMON_DATASET = ["Udacity"]
COMMON_TYPE = "eval"

# COMMON_DATA_ROOT = "/raid/007--Experiments/selforacle/evaluation_data"
COMMON_DATA_ROOT =  "/raid/007--Experiments/selforacle/drive_dataset/test/normal"
#ATTACKED_DATA = "/raid/007--Experiments/selforacle/evaluation/pgd_attack/epoch"  # "fgsm" or "pgd"
COMMON_ATTACK_TYPE = ["weather","weather","weather","attack","attack","attack","attack","attack","attack"]
#COMMON_ATTACK_TYPE = ["attack"]
# COMMON_ATTACK_DATA_ROOT = "/raid/007--Experiments/selforacle/A-Unified-Benchmark-for-Out-of-Distribution-Detection-for-Autonomous-Driving-Systems/DataPreprocess/images_out/spsa_attack_epoch"
# COMMON_ATTACK_DATA_ROOT = "/raid/007--Experiments/selforacle/evaluation/fgsm_attack/dave2v1"
# COMMON_ATTACK_DATA_ROOT = [["/raid/007--Experiments/selforacle/evaluation_data","CHAUFFEUR-Track1-Fog"],
#                            ["/raid/007--Experiments/selforacle/evaluation_data","CHAUFFEUR-Track1-Rain"],
#                            ["/raid/007--Experiments/selforacle/evaluation_data","CHAUFFEUR-Track1-Snow"],
#                            "/raid/007--Experiments/selforacle/evaluation/fgsm_attack/dave2v1",
#                            "/raid/007--Experiments/selforacle/evaluation/fgsm_attack/epoch",
#                            "/raid/007--Experiments/selforacle/evaluation/pgd_attack/dave2v1",
#                            "/raid/007--Experiments/selforacle/evaluation/pgd_attack/epoch",
#                            "/raid/007--Experiments/selforacle/evaluation/sp_attack/epoch",
#                            "/raid/007--Experiments/selforacle/evaluation/spsa_attack/epoch",
#                            ]
COMMON_ATTACK_DATA_ROOT = ["/raid/007--Experiments/selforacle/drive_dataset/test/fog",
                           "/raid/007--Experiments/selforacle/drive_dataset/test/rain",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/snow",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/fgsm_attack_dave2v1",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/fgsm_attack_epoch",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/pgd_attack_dave2v1",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/pgd_attack_epoch",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/sp_attack_epoch",
                            "/raid/007--Experiments/selforacle/drive_dataset/test/spsa_attack_epoch",    
                            ]
# COMMON_ATTACK_DATA_ROOT = [ 
#                             "/raid/007--Experiments/selforacle/evaluation/spsa_attack/epoch",
#                            ]

COMMON_THRESHOLD = [[70],[80],[90],[95],[99]]