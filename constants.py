
DATA_DIR                = "data/"
FILE_TRAIN_LABELED      = DATA_DIR + "train_labeled.p"
FILE_TRAIN_LABELED_AUG  = DATA_DIR + "train_labeled_aug.p"
FILE_TRAIN_UNLABELED    = DATA_DIR + "train_unlabeled.p"
FILE_VALIDATION         = DATA_DIR + "validation.p"
FILE_TEST               = DATA_DIR + "test.p"

MODEL_DIR               = "model/"
MODEL_NAME              = MODEL_DIR + "network.p"

#   LADDER
MEAN = 0
STD  = 0.3		

lam = [10, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

Xtranslate = 2
Ytranslate = 2
Rotate     = 30
NUM_JITTERS = 5