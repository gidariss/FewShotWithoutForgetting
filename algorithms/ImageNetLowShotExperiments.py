from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils

from pdb import set_trace as breakpoint
import time
import h5py
from tqdm import tqdm

from . import Algorithm
from . import FewShot


def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def softmax_with_novel_prior(scores, novel_inds, base_inds, prior_m):
    scores = torch.exp(scores)
    scores_novel = scores[:, novel_inds]
    scores_base = scores[:, base_inds]
    tol = 0.0000001
    scores_novel *= prior_m / (tol + torch.sum(scores_novel, dim=1, keepdim=True).expand_as(scores_novel))
    scores_base *= (1.0 - prior_m) / (tol + torch.sum(scores_base, dim=1, keepdim=True).expand_as(scores_base))
    scores[:, novel_inds] = scores_novel
    scores[:, base_inds] = scores_base
    return scores


class ImageNetLowShotExperiments(FewShot):
    def __init__(self, opt):
        FewShot.__init__(self, opt)

    def save_features(self, dataloader, filename):
        """Saves features and labels for each image in the dataloader.

        This routines uses the trained feature model (i.e.,
        self.networks['feat_model']) in order to extract a feature for each
        image in the dataloader. The extracted features along with the labels
        of the images that they come from are saved in a h5py file.

        Args:
            dloader: A dataloader that feeds images and labels.
            filename: The file name where the features and the labels of each
                images in the dataloader are saved.
        """
        feat_model = self.networks['feat_model']
        feat_model.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info('Destination filename for features: {0}'.format(
            filename))
        data_file = h5py.File(filename, 'w')
        max_count = len(dataloader_iterator) * dataloader_iterator.batch_size
        all_labels = data_file.create_dataset(
            'all_labels', (max_count,), dtype='i')
        all_features = None

        count = 0
        for i, batch in enumerate(tqdm(dataloader_iterator)):
            images, labels = batch
            self.tensors['images_test'].resize_(images.size()).copy_(images)
            self.tensors['labels_test'].resize_(labels.size()).copy_(labels)
            images = self.tensors['images_test']
            labels = self.tensors['labels_test']
            assert(images.dim()==4 and labels.dim()==1)
            features = feat_model(Variable(images, volatile=True))
            assert(features.dim()==2)

            if all_features is None:
                self.logger.info('Image size: {0}'.format(images.size()))
                self.logger.info('Feature size: {0}'.format(features.size()))
                self.logger.info('Max_count: {0}'.format(max_count))
                all_features = data_file.create_dataset(
                    'all_features', (max_count, features.size(1)), dtype='f')
                self.logger.info('Number of feature channels: {0}'.format(
                    features.size(1)))

            all_features[count:(count + features.size(0)), :] = (
                features.data.cpu().numpy())
            all_labels[count:(count + features.size(0))] = labels.cpu().numpy()
            count = count + features.size(0)

        self.logger.info('Number of processed primages: {0}'.format(count))

        count_var = data_file.create_dataset('count', (1,), dtype='i')
        count_var[0] = count
        data_file.close()


    def preprocess_novel_training_data(self, nove_cat_training_data):
        """Preprocess the novel training data."""

        images_train, labels_train, Kids, nKbase, nKnovel = nove_cat_training_data
        self.nKbase = nKbase
        self.nKnovel = nKnovel

        # Insert an extra singleton dimension.
        images_train = images_train.unsqueeze(dim=0)
        labels_train = labels_train.unsqueeze(dim=0)
        Kids = Kids.unsqueeze(dim=0)

        self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
        self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
        self.tensors['Kids'].resize_(Kids.size()).copy_(Kids)
        labels_train = self.tensors['labels_train']

        labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
        dim = len(labels_train_1hot_size) - 1
        labels_train = labels_train.unsqueeze(dim=labels_train.dim())
        self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
            dim,labels_train - nKbase, 1)

    def add_novel_categories(self, nove_cat_training_data):
        """Add the training data of the novel categories to the model."""

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        feat_model.eval()
        classifier.eval()

        self.preprocess_novel_training_data(nove_cat_training_data)
        nKbase = self.nKbase
        nKnovel = self.nKnovel

        images = self.tensors['images_train']
        labels_train_1hot = self.tensors['labels_train_1hot']
        Kids = self.tensors['Kids']

        #***********************************************************************
        #*********************** SET TORCH VARIABLES ***************************
        Kbase_ids_var = Variable(
            Kids[:,:nKbase].contiguous(), requires_grad=False)
        labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False)
        images_var = Variable(images, volatile=True)
        #***********************************************************************

        #***********************************************************************
        #******************** EXTRACT FEATS FROM EXEMPLARS *********************
        batch_size, num_train_examples, channels, height, width = images_var.size()
        features_train = feat_model(images_var.view(
            batch_size * num_train_examples, channels, height, width))
        features_train = features_train.view(
            [batch_size, num_train_examples,] + list(features_train.size()[1:]))
        #***********************************************************************
        #******************** GET CLASSIFICATION WEIGHTS ***********************
        # The following routine returns the classification weight vectors of
        # both the base and then novel categories. For the novel categories,
        # the classification weight vectors are generated using the training
        # features for those novel cateogories.
        clsWeights = classifier.get_classification_weights(
            Kbase_ids_var, features_train, labels_train_1hot_var)
        #***********************************************************************
        self.tensors['clsWeights'] = clsWeights.data.clone()


    def evaluate_model_on_test_images(
        self, data_loader, base_classes, novel_classes, exp_id='', prior_m=0.8):
        """Evaluate the model.

        It is assumed that the user has already called the routine
        add_novel_categories() before calling this function.

        Args:
            data_loader: data loader that feeds test images and lables in order
                to evaluatethe model.
            base_classes: A list with the labels of the base categories that
                will be used for evaluation.
            novel_classes: A list with the labels of the novel categories that
                will be used for evaluation.
            exp_id: A string with the id of the experiment.
            prior_m: A scalar in the range [0, 1.0] that represents the prior
                for whether a test image comes from the novel / base categories.
        """

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        feat_model.eval()
        classifier.eval()
        clsWeights = self.tensors['clsWeights']

        both_classes = base_classes + novel_classes
        # Not valid classes are those that do not belong neighter to the base
        # nor the nor the novel classes.
        nKall = self.nKbase + self.nKnovel
        not_valid_classes = list(set(range(nKall)).difference(set(both_classes)))
        not_valid_classes_torch = torch.Tensor(not_valid_classes).long().cuda()
        base_classes_torch = torch.Tensor(base_classes).long().cuda()
        novel_classes_torch = torch.Tensor(novel_classes).long().cuda()

        top1, top1_novel, top1_base, top1_prior = None, None, None, None
        top5, top5_novel, top5_base, top5_prior = None, None, None, None
        all_labels = None
        for idx, batch in enumerate(tqdm(data_loader(0))):
            images_test, labels_test = batch
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            images_test = self.tensors['images_test']
            labels_test = self.tensors['labels_test']

            images_test_var = Variable(images_test, volatile=True)
            labels_var = Variable(labels_test, volatile=True)
            clsWeights_var = Variable(clsWeights, volatile=True)
            num_test_examples = images_test_var.size(0)

            features_var = feat_model(images_test_var).view(1, num_test_examples, -1)
            scores_var = classifier.apply_classification_weights(
                features_var, clsWeights_var).view(num_test_examples, -1)

            scores = scores_var.data
            scores_prior = softmax_with_novel_prior(
                scores.clone(), novel_classes_torch, base_classes_torch, prior_m)

            scores[:, not_valid_classes_torch] = -1000
            top1_this, top5_this = compute_top1_and_top5_accuracy(scores, labels_test)
            top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
            top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))

            scores_prior[:, not_valid_classes_torch] = -1000
            top1_this, top5_this = compute_top1_and_top5_accuracy(scores_prior, labels_test)
            top1_prior = top1_this if top1_prior is None else np.concatenate((top1_prior, top1_this))
            top5_prior = top5_this if top5_prior is None else np.concatenate((top5_prior, top5_this))

            scores_novel = scores.clone()
            scores_novel[:, base_classes_torch] = -1000
            top1_this, top5_this = compute_top1_and_top5_accuracy(scores_novel, labels_test)
            top1_novel = top1_this if top1_novel is None else np.concatenate((top1_novel, top1_this))
            top5_novel = top5_this if top5_novel is None else np.concatenate((top5_novel, top5_this))

            scores_base = scores.clone()
            scores_base[:, novel_classes_torch] = -1000
            top1_this, top5_this = compute_top1_and_top5_accuracy(scores_base, labels_test)
            top1_base = top1_this if top1_base is None else np.concatenate((top1_base, top1_this))
            top5_base = top5_this if top5_base is None else np.concatenate((top5_base, top5_this))

            labels_test_np = labels_test.cpu().numpy()
            all_labels = labels_test_np if all_labels is None else np.concatenate((all_labels, labels_test_np))

        is_novel = np.in1d(all_labels, np.array(novel_classes))
        is_base = np.in1d(all_labels, np.array(base_classes))
        is_either = is_novel | is_base

        top1_novel = 100*np.mean(top1_novel[is_novel])
        top1_base = 100*np.mean(top1_base[is_base])
        top1_all = 100*np.mean(top1[is_either])
        top1_all_prior = 100*np.mean(top1_prior[is_either])

        top5_novel = 100*np.mean(top5_novel[is_novel])
        top5_base = 100*np.mean(top5_base[is_base])
        top5_all = 100*np.mean(top5[is_either])
        top5_all_prior = 100*np.mean(top5_prior[is_either])

        self.logger.info('Experiment {0}'.format(exp_id))
        self.logger.info(
            '==> Top 5 Accuracies: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | All prior {3:3.2f}]'
            .format(top5_novel, top5_base, top5_all, top5_all_prior))
        self.logger.info(
            '==> Top 1 Accuracies: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | All prior {3:3.2f}]'
            .format(top1_novel, top1_base, top1_all, top1_all_prior))

        results_array = np.array(
            [top5_novel, top5_base, top5_all, top5_all_prior,
             top1_novel, top1_base, top1_all, top1_all_prior]).reshape(1,-1)

        return results_array

    def lowshot_avg_results(self, results_all, exp_id=''):
        results_all = np.concatenate(results_all, axis=0)
        num_eval_experiments = results_all.shape[0]

        mu_results = results_all.mean(axis=0)
        top5_novel = mu_results[0]
        top5_base = mu_results[1]
        top5_all = mu_results[2]
        top5_all_prior = mu_results[3]
        top1_novel = mu_results[4]
        top1_base = mu_results[5]
        top1_all = mu_results[6]
        top1_all_prior = mu_results[7]

        std_results  = results_all.std(axis=0)
        ci95_results = 1.96*std_results/np.sqrt(results_all.shape[0])

        top5_novel_ci95 = ci95_results[0]
        top5_base_ci95 = ci95_results[1]
        top5_all_ci95 = ci95_results[2]
        top5_all_prior_ci95 = ci95_results[3]
        top1_novel_ci95 = ci95_results[4]
        top1_base_ci95 = ci95_results[5]
        top1_all_ci95 = ci95_results[6]
        top1_all_prior_ci95 = ci95_results[7]

        self.logger.info('----------------------------------------------------------------')
        self.logger.info('Average results of {0} experiments: {1}'.format(
            num_eval_experiments, exp_id))
        self.logger.info(
            '==> Top 5 Accuracies:      [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | All prior {3:3.2f}]'
            .format(top5_novel, top5_base, top5_all, top5_all_prior))
        self.logger.info(
            '==> Top 5 conf. intervals: [Novel: {0:5.2f} | Base: {1:5.2f} | All {2:5.2f} | All prior {3:5.2f}]'
            .format(top5_novel_ci95, top5_base_ci95, top5_all_ci95, top5_all_prior_ci95))
        self.logger.info('----------------------------------------------------------------')
        self.logger.info(
            '==> Top 1 Accuracies:      [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | All prior {3:3.2f}]'
            .format(top1_novel, top1_base, top1_all, top1_all_prior))
        self.logger.info(
            '==> Top 1 conf. intervals: [Novel: {0:5.2f} | Base: {1:5.2f} | All {2:5.2f} | All prior {3:5.2f}]'
            .format(top1_novel_ci95, top1_base_ci95, top1_all_ci95, top1_all_prior_ci95))
        self.logger.info('----------------------------------------------------------------')
