import os

EPOCHS = 100
BATCH_SIZE = 8
FEATURES_SHAPE = (250, 6)

# Optimizer_lr_activation_epochs
rootdir = "adam_0001_ELU_100"
if os.path.isdir(rootdir):
	rootdir = rootdir + '_v2'
	i = 3
	while os.path.isdir(rootdir):
		rootdir = rootdir[:-1] + str(i)
		i += 1

logdir = os.path.join(rootdir, "log")

databin = "databin"
os.makedirs(databin, exist_ok=True)

model_dir = os.path.join(rootdir, "saved_model")
os.makedirs(model_dir, exist_ok=True)
ckpt = os.path.join(model_dir, "model.ckpt")