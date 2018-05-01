import os

EPOCHS = 100
BATCH_SIZE = 8
FEATURES_SHAPE = (250, 6)

logdir = "log"

databin = "databin"
os.makedirs(databin, exist_ok=True)

model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)
ckpt = os.path.join(model_dir, "model.ckpt")