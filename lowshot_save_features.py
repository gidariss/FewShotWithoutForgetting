"""Extracts and saves features (with a model trained by the lowshot_train_stage1.py routine) from the
images of the ImageNet dataset.

Example of usage:
# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=train

The config argument specifies the model that will be used.
"""

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import ImageNet, SimpleDataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    default="",
    help="config file with hyper-parameters of the model that we will use for "
    "extracting features from ImageNet dataset.",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    default=-1,
    help="checkpoint (epoch id) that will be loaded. If a negative value is"
    " given then the latest existing checkpoint is loaded.",
)
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument("--split", type=str, default="val")
args_opt = parser.parse_args()

exp_config_file = os.path.join(".", "config", args_opt.config + ".py")
exp_directory = os.path.join(".", "experiments", args_opt.config)

# Load the configuration params of the experiment
print("Launching experiment: %s" % exp_config_file)
config = imp.load_source("", exp_config_file).config
config["exp_dir"] = exp_directory
print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
print(
    "Generated logs, snapshots, and config files will be stored on %s"
    % (config["exp_dir"])
)

if (args_opt.split != "train") and (args_opt.split != "val"):
    raise ValueError("Not valid split {0}".format(args_opt.split))

dataset = ImageNet(split=args_opt.split)
dloader = SimpleDataloader(dataset, batch_size=256)

algorithm = alg.ImageNetLowShotExperiments(config)

if args_opt.cuda:  # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0:  # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*", train=False
    )

dst_directory = os.path.join(".", "data", "IMAGENET", args_opt.config)
if not os.path.isdir(dst_directory):
    os.makedirs(dst_directory)
dst_filename = os.path.join(
    dst_directory, "feature_dataset_" + args_opt.split + ".json"
)

algorithm.save_features(dloader, dst_filename)
