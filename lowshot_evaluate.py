"""Evaluates a fewshot recognition models on the low-shot Imagenet dataset[*]
using the improved evaluation metrics proposed by Wang et al [**].

Example of usage:
# Evaluate the model for the 1-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN1 --testset
# Evaluate the model for the 2-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN2 --testset
# Evaluate the model for the 5-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN5 --testset
# Evaluate the model for the 10-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN10 --testset
# Evaluate the model for the 20-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN20 --testset

The config argument specifies the model that will be evaluated.

[*] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[**] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
"""

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import ImageNetLowShotFeatures, LowShotDataloader

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
    default=-1,
    help="checkpoint (epoch id) that will be loaded. If a negative value is "
    "given then the latest existing checkpoint is loaded.",
)
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument(
    "--testset",
    default=False,
    action="store_true",
    help="If True, the model is evaluated on the test set of ImageNetLowShot. "
    "Otherwise, the validation set is used for evaluation.",
)
parser.add_argument(
    "--num_exp",
    type=int,
    default=100,
    help="the number of evaluation experiments that will run before computing "
    "the average performance.",
)
parser.add_argument("--prior", type=float, default=0.8)
args_opt = parser.parse_args()

exp_config_file = os.path.join(".", "config", args_opt.config + ".py")
exp_directory = os.path.join(".", "experiments", args_opt.config)

# Load the configuration params of the experiment.
print("Launching experiment: %s" % exp_config_file)
config = imp.load_source("", exp_config_file).config
config["exp_dir"] = exp_directory
print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
print(
    "Generated logs, snapshots, and model files will be stored on %s"
    % (config["exp_dir"])
)

algorithm = alg.ImageNetLowShotExperiments(config)
if args_opt.cuda:  # enable cuda.
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0:  # load checkpoint.
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else "*", train=False
    )

# Prepare the datasets and the the dataloader.
nExemplars = data_train_opt = config["data_train_opt"]["nExemplars"]
eval_phase = "test" if args_opt.testset else "val"

feat_data_train = ImageNetLowShotFeatures(
    data_dir=config["data_dir"], image_split="train", phase=eval_phase
)
feat_data_test = ImageNetLowShotFeatures(
    data_dir=config["data_dir"], image_split="val", phase=eval_phase
)
data_loader = LowShotDataloader(
    feat_data_train,
    feat_data_test,
    nExemplars=nExemplars,
    batch_size=26000,
    num_workers=1,
)

results = []
# Run args_opt.num_exp different number of evaluation experiments (each time
# sampling a different set of training images for the the novel categories).
for exp_id in range(args_opt.num_exp):
    # Sample training data for the novel categories from the training set of
    # ImageNet.
    # nove_cat_data = dloader.getNovelCategoriesTrainingData(exp_id=exp_id)
    nove_cat_data = data_loader.sample_training_data_for_novel_categories(exp_id=exp_id)
    # Feed the training data of the novel categories to the algorithm.
    algorithm.add_novel_categories(nove_cat_data)
    # Evaluate on the validation images of ImageNet.
    results_this = algorithm.evaluate_model_on_test_images(
        data_loader=data_loader,
        base_classes=data_loader.base_category_label_indices(),
        novel_classes=data_loader.novel_category_label_indices(),
        exp_id="Exp_id = " + str(exp_id),
        prior_m=args_opt.prior,
    )
    results.append(results_this)

# Print the average results.
algorithm.lowshot_avg_results(
    results,
    exp_id=args_opt.config
    + " nExemplars = "
    + str(nExemplars)
    + " prior = "
    + str(args_opt.prior),
)
