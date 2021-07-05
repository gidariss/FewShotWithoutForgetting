"""Applies the 2nd training stage of our approach on the low-shot Imagenet dataset[*].

Example of usage:
# Training the model for the 1-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN1
# Training the model for the 2-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN2
# Training the model for the 5-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN5
# Training the model for the 10-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN10
# Training the model for the 20-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN20

All the configuration files above (i.e., specified by the --config argument) are placed on the
directory ./config .

[*] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
"""

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import ImageNetLowShotFeatures, FewShotDataloader, LowShotDataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    default="",
    help="config file with parameters of the experiment",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    default=0,
    help="checkpoint (epoch id) that will be loaded. If a negative value is "
    "given then the latest existing checkpoint is loaded.",
)
parser.add_argument(
    "--num_workers", type=int, default=4, help="number of data loading workers"
)
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument(
    "--disp_step", type=int, default=200, help="display step during training"
)
args_opt = parser.parse_args()

exp_config_file = os.path.join(".", "config", args_opt.config + ".py")
exp_directory = os.path.join(".", "experiments", args_opt.config)

# Load the configuration params of the experiment
print("Launching experiment: %s" % exp_config_file)
config = imp.load_source("", exp_config_file).config
config[
    "exp_dir"
] = exp_directory  # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
print(
    "Generated logs, snapshots, and model files will be stored on %s"
    % (config["exp_dir"])
)

config["disp_step"] = args_opt.disp_step
algorithm = alg.ImageNetLowShotExperiments(config)
if args_opt.cuda:  # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0:  # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*", train=True
    )

# Set the train dataset and the corresponding data loader.
data_train_opt = config["data_train_opt"]
feat_dataset_train = ImageNetLowShotFeatures(
    data_dir=config["data_dir"], image_split="train", phase="train"
)
dloader_train = FewShotDataloader(
    dataset=feat_dataset_train,
    nKnovel=data_train_opt["nKnovel"],
    nKbase=data_train_opt["nKbase"],
    nExemplars=data_train_opt["nExemplars"],  # num training examples per novel category
    nTestNovel=data_train_opt[
        "nTestNovel"
    ],  # num test examples for all the novel categories
    nTestBase=data_train_opt[
        "nTestBase"
    ],  # num test examples for all the base categories
    batch_size=data_train_opt["batch_size"],
    num_workers=args_opt.num_workers,
    epoch_size=data_train_opt["epoch_size"],  # num of batches per epoch
)

algorithm.solve(dloader_train)
