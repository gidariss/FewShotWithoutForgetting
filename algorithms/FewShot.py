from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils

from pdb import set_trace as breakpoint
import time
from tqdm import tqdm

from . import Algorithm


def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


def activate_dropout_units(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = True


class FewShot(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.nKbase = torch.LongTensor()
        self.activate_dropout = (
            opt['activate_dropout'] if ('activate_dropout' in opt) else False)
        self.keep_best_model_metric_name = 'AccuracyNovel'

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images_train'] = torch.FloatTensor()
        self.tensors['labels_train'] = torch.LongTensor()
        self.tensors['labels_train_1hot'] = torch.FloatTensor()
        self.tensors['images_test'] = torch.FloatTensor()
        self.tensors['labels_test'] = torch.LongTensor()
        self.tensors['Kids'] = torch.LongTensor()

    def set_tensors(self, batch):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel

        if self.nKnovel > 0:
            train_test_stage = 'fewshot'
            assert(len(batch) == 6)
            images_train, labels_train, images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0]
            self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
            self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors['labels_train']

            nKnovel = 1 + labels_train.max() - self.nKbase

            labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1)
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)
        else:
            train_test_stage = 'base_classification'
            assert(len(batch) == 4)
            images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0]
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train):
        process_type = self.set_tensors(batch)

        if process_type=='fewshot':
            record = self.process_batch_fewshot_without_forgetting(
                do_train=do_train)
        elif process_type=='base_classification':
            record = self.process_batch_base_category_classification(
                do_train=do_train)
        else:
            raise ValueError('Unexpected process type {0}'.format(process_type))

        return record

    def process_batch_base_category_classification(self, do_train=True):
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        criterion  = self.criterions['loss']
        if do_train: # zero the gradients
            self.optimizers['feat_model'].zero_grad()
            self.optimizers['classifier'].zero_grad()
        #********************************************************

        #***********************************************************************
        #*********************** SET TORCH VARIABLES ***************************
        images_test_var = Variable(images_test, volatile=(not do_train))
        labels_test_var = Variable(labels_test, requires_grad=False)
        Kbase_var = (None if (nKbase==0) else Variable(
            Kids[:,:nKbase].contiguous(),requires_grad=False))
        #***********************************************************************

        loss_record  = {}
        #***********************************************************************
        #************************* FORWARD PHASE *******************************
        #*********** EXTRACT FEATURES FROM TRAIN & TEST IMAGES *****************
        batch_size, num_test_examples, channels, height, width = images_test.size()
        new_batch_dim = batch_size * num_test_examples
        features_test_var = feat_model(
            images_test_var.view(new_batch_dim, channels, height, width))
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:]))
        #************************ APPLY CLASSIFIER *****************************
        cls_scores_var = classifier(features_test=features_test_var, Kbase_ids=Kbase_var)
        cls_scores_var = cls_scores_var.view(new_batch_dim,-1)
        labels_test_var = labels_test_var.view(new_batch_dim)
        #***********************************************************************
        #************************** COMPUTE LOSSES *****************************
        loss_cls_all = criterion(cls_scores_var, labels_test_var)
        loss_total = loss_cls_all
        loss_record['loss'] = loss_total.data[0]
        loss_record['AccuracyBase'] = top1accuracy(
            cls_scores_var.data, labels_test_var.data)
        #***********************************************************************

        #***********************************************************************
        #************************* BACKWARD PHASE ******************************
        if do_train:
            loss_total.backward()
            self.optimizers['feat_model'].step()
            self.optimizers['classifier'].step()
        #***********************************************************************

        return loss_record

    def process_batch_fewshot_without_forgetting(self, do_train=True):
        images_train = self.tensors['images_train']
        labels_train = self.tensors['labels_train']
        labels_train_1hot = self.tensors['labels_train_1hot']
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        criterion = self.criterions['loss']

        do_train_feat_model = do_train and self.optimizers['feat_model'] is not None
        if (not do_train_feat_model):
            feat_model.eval()
            if do_train and self.activate_dropout:
                # Activate the dropout units of the feature extraction model
                # even if the feature extraction model is freezed (i.e., it is
                # in eval mode).
                activate_dropout_units(feat_model)

        if do_train: # zero the gradients
            if do_train_feat_model:
                self.optimizers['feat_model'].zero_grad()
            self.optimizers['classifier'].zero_grad()

        #***********************************************************************
        #*********************** SET TORCH VARIABLES ***************************
        is_volatile = (not do_train or not do_train_feat_model)
        images_test_var = Variable(images_test, volatile=is_volatile)
        labels_test_var = Variable(labels_test, requires_grad=False)
        Kbase_var = (None if (nKbase==0) else
            Variable(Kids[:,:nKbase].contiguous(),requires_grad=False))
        labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False)
        images_train_var = Variable(images_train, volatile=is_volatile)
        #***********************************************************************

        loss_record = {}
        #***********************************************************************
        #************************* FORWARD PHASE: ******************************

        #************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        features_train_var = feat_model(
            images_train_var.view(batch_size * num_train_examples, channels, height, width)
        )
        features_test_var = feat_model(
            images_test_var.view(batch_size * num_test_examples, channels, height, width)
        )
        features_train_var = features_train_var.view(
            [batch_size, num_train_examples,] + list(features_train_var.size()[1:])
        )
        features_test_var = features_test_var.view(
            [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
        )
        if (not do_train_feat_model) and do_train:
            # Make sure that no gradients are backproagated to the feature
            # extractor when the feature extraction model is freezed.
            features_train_var = Variable(features_train_var.data, volatile=False)
            features_test_var = Variable(features_test_var.data, volatile=False)
        #***********************************************************************

        #************************ APPLY CLASSIFIER *****************************
        if self.nKbase > 0:
            cls_scores_var = classifier(
                features_test=features_test_var,
                Kbase_ids=Kbase_var,
                features_train=features_train_var,
                labels_train=labels_train_1hot_var)
        else:
            cls_scores_var = classifier(
                features_test=features_test_var,
                features_train=features_train_var,
                labels_train=labels_train_1hot_var)

        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)
        labels_test_var = labels_test_var.view(batch_size * num_test_examples)
        #***********************************************************************

        #************************* COMPUTE LOSSES ******************************
        loss_cls_all = criterion(cls_scores_var, labels_test_var)
        loss_total = loss_cls_all
        loss_record['loss'] = loss_total.data[0]

        if self.nKbase > 0:
            loss_record['AccuracyBoth'] = top1accuracy(
                cls_scores_var.data, labels_test_var.data)

            preds_data = cls_scores_var.data.cpu()
            labels_test_data = labels_test_var.data.cpu()
            base_ids = torch.nonzero(labels_test_data < self.nKbase).view(-1)
            novel_ids = torch.nonzero(labels_test_data >= self.nKbase).view(-1)
            preds_base = preds_data[base_ids,:]
            preds_novel = preds_data[novel_ids,:]

            loss_record['AccuracyBase'] = top1accuracy(
                preds_base[:,:nKbase], labels_test_data[base_ids])
            loss_record['AccuracyNovel'] = top1accuracy(
                preds_novel[:,nKbase:], (labels_test_data[novel_ids]-nKbase))
        else:
            loss_record['AccuracyNovel'] = top1accuracy(
                cls_scores_var.data, labels_test_var.data)
        #***********************************************************************
        
        #***********************************************************************
        #************************* BACKWARD PHASE ******************************
        if do_train:
            loss_total.backward()
            if do_train_feat_model:
                self.optimizers['feat_model'].step()
            self.optimizers['classifier'].step()
        #***********************************************************************

        if (not do_train):
            if self.biter == 0: self.test_accuracies = {'AccuracyNovel':[]}
            self.test_accuracies['AccuracyNovel'].append(
                loss_record['AccuracyNovel'])
            if self.biter == (self.bnumber - 1):
                # Compute the std and the confidence interval of the accuracy of
                # the novel categories.
                stds = np.std(np.array(self.test_accuracies['AccuracyNovel']), 0)
                ci95 = 1.96*stds/np.sqrt(self.bnumber)
                loss_record['AccuracyNovel_std'] = stds
                loss_record['AccuracyNovel_cnf'] = ci95

        return loss_record
