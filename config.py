"""
@author: Ankit Bindal
"""

import os

print("Change config.py to change hyperparameters.")

EPOCHS = 150
BATCH_SIZE = 8
RANDOM_SEED = 7
FEATURES_SHAPE = (500, 6)

# Directory for storing binary version of dataset.
databin = "databin"
os.makedirs(databin, exist_ok=True)

optimizer = "adam"
lr = 0.0001
activation = "elu"

# Optimizer_lr_activation_epochs
strLR = str(lr)[str(lr).index('.')+1:]
rootdir = os.path.join("output", 
						optimizer+"_"+strLR+"_"+activation+"_"+str(EPOCHS)+"_v1")

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
