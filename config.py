"""
@author: Ankit Bindal
"""

import os

print("Change config.py to change hyperparameters.")

EPOCHS = 100
BATCH_SIZE = 4
RANDOM_SEED = 5
FEATURES_SHAPE = (500, 6)

optimizer = "adam"
lr = 0.0001
activation = "elu"
resume_ckpt = os.path.join("output", "adam_0001_elu_100_v1", "saved_model", "model.ckpt-95")

# Directory for storing binary version of dataset.
databin = "databin"
os.makedirs(databin, exist_ok=True)

# Optimizer_lr_activation_epochs_version
strLR = str(lr)[str(lr).index('.')+1:]
rootdir = os.path.join("output", "{}_{}_{}_{}_v1".format(optimizer, strLR, activation, EPOCHS))

# Rename the directory if it exists.
if os.path.isdir(rootdir):
	rootdir = rootdir[: rootdir.rfind("_")+2] + '2'
	i = 3
	while os.path.isdir(rootdir):
		rootdir = rootdir[: rootdir.rfind("_")+2]
		rootdir = rootdir + str(i)
		i += 1

logdir = os.path.join(rootdir, "log")
model_dir = os.path.join(rootdir, "saved_model")
ckpt = os.path.join(model_dir, "model.ckpt")

def init():
	os.makedirs(model_dir, exist_ok=True)
	print("Saving to {}".format(model_dir))

	# Write config settings to file
	setting = "Optimizer: {}\n".format(optimizer)
	setting += "Activation: {}\n".format(activation)
	setting += "Learning rate: {}\n".format(lr)
	setting += "Epochs: {}\n".format(EPOCHS)
	setting += "Features shape: {}\n".format(FEATURES_SHAPE)
	setting += "Batch size: {}\n".format(BATCH_SIZE)
	
	with open(os.path.join(rootdir, "config.txt"), 'w') as f:
		f.write(setting)
