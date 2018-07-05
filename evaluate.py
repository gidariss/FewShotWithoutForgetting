"""Evaluates a fewshot recognition models on MiniImagenet.

The routines evaluates the model that has achieved the best so far accuracy on
the recognition of novel categories on the validation set of MiniImagenet.

Example of usage on evaluating the proposed approach model already trained with
train routine:

# Evaluating the model for the 1-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 --testset

# Evaluating the model for the 5-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 --testset

The config argument specifies the model that will be evaluated.
"""

from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import MiniImageNet, FewShotDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment. It is assumed that all'
         ' the config file is placed on  ')
parser.add_argument('--evaluate', default=False, action='store_true',
    help='If True, then no training is performed and the model is only '
         'evaluated on the validation or test set of MiniImageNet.')
parser.add_argument('--num_workers', type=int, default=4,
    help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--testset', default=False, action='store_true',
    help='If True, the model is evaluated on the test set of MiniImageNet. '
         'Otherwise, the validation set is used for evaluation.')
args_opt = parser.parse_args()

exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
exp_directory = os.path.join('.', 'experiments', args_opt.config)

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print('Loading experiment %s from file: %s' %
      (args_opt.config, exp_config_file))
print('Generated logs, snapshots, and model files will be stored on %s' %
      (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
if args_opt.testset:
    test_split = 'test'
    epoch_size = 600
else:
    test_split = 'val'
    epoch_size = 2000

nExemplars = config['data_test_opt']['nExemplars']
dloader_test = FewShotDataloader(
    dataset=MiniImageNet(phase=test_split),
    nKnovel=5, # number of novel categories on each training episode.
    nKbase=64, # number of base categories.
    nExemplars=nExemplars, # num training examples per novel category
    nTestNovel=15 * 5, # num test examples for all the novel categories
    nTestBase=15 * 5, # num test examples for all the base categories
    batch_size=1,
    num_workers=0,
    epoch_size=epoch_size, # num of batches per epoch
)

algorithm = alg.FewShot(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

# In evaluation mode we load the checkpoint with the highest novel category
# recognition accuracy on the validation set of MiniImagenet.
algorithm.load_checkpoint(epoch='*', train=False, suffix='.best')

# Run evaluation.
algorithm.evaluate(dloader_test)
