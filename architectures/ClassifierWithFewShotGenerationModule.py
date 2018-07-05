import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as breakpoint


class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1,2)
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel


class AttentionBasedBlock(nn.Module):
    def __init__(self, nFeat, nK, scale_att=10.0):
        super(AttentionBasedBlock, self).__init__()
        self.nFeat = nFeat
        self.queryLayer = nn.Linear(nFeat, nFeat)
        self.queryLayer.weight.data.copy_(
            torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat)*0.001)
        self.queryLayer.bias.data.zero_()

        self.scale_att = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_att), requires_grad=True)
        wkeys = torch.FloatTensor(nK, nFeat).normal_(0.0, np.sqrt(2.0/nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)


    def forward(self, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1) # [batch_size x nKbase x num_features]
        labels_train_transposed = labels_train.transpose(1,2)
        nKnovel = labels_train_transposed.size(1) # [batch_size x nKnovel x num_train_examples]

        features_train = features_train.view(
            batch_size*num_train_examples, num_features)
        Qe = self.queryLayer(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim()-1, eps=1e-12)

        wkeys = self.wkeys[Kbase.view(-1)] # the keys of the base categoreis
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim()-1, eps=1e-12)
        # Transpose from [batch_size x nKbase x nFeat] to
        # [batch_size x self.nFeat x nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1,2)

        # Compute the attention coeficients
        # batch matrix multiplications: AttentionCoeficients = Qe * wkeys ==>
        # [batch_size x num_train_examples x nKbase] =
        #   [batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoeficients = self.scale_att * torch.bmm(Qe, wkeys)
        AttentionCoeficients = F.softmax(
            AttentionCoeficients.view(batch_size*num_train_examples, nKbase))
        AttentionCoeficients = AttentionCoeficients.view(
            batch_size, num_train_examples, nKbase)

        # batch matrix multiplications: weight_novel = AttentionCoeficients * weight_base ==>
        # [batch_size x num_train_examples x num_features] =
        #   [batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoeficients, weight_base)
        # batch matrix multiplications: weight_novel = labels_train_transposed * weight_novel ==>
        # [batch_size x nKnovel x num_features] =
        #   [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))

        return weight_novel


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.weight_generator_type = opt['weight_generator_type']
        self.classifier_type = opt['classifier_type']
        assert(self.classifier_type == 'cosine' or
               self.classifier_type == 'dotproduct')

        nKall = opt['nKall']
        nFeat = opt['nFeat']
        self.nFeat = nFeat
        self.nKall = nKall

        weight_base = torch.FloatTensor(nKall, nFeat).normal_(
            0.0, np.sqrt(2.0/nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        self.bias = nn.Parameter(
            torch.FloatTensor(1).fill_(0), requires_grad=True)
        scale_cls = opt['scale_cls'] if ('scale_cls' in opt) else 10.0
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls),
            requires_grad=True)

        if self.weight_generator_type == 'none':
            # If the weight generator type is `none` then feature averaging
            # is being used. However, in this case the generator does not
            # involve any learnable parameter and thus does not require
            # training.
            self.favgblock = FeatExemplarAvgBlock(nFeat)
        elif self.weight_generator_type=='feature_averaging':
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.wnLayerFavg = LinearDiag(nFeat)
        elif self.weight_generator_type=='attention_based':
            scale_att = opt['scale_att'] if ('scale_att' in opt) else 10.0
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.attblock = AttentionBasedBlock(
                nFeat, nKall, scale_att=scale_att)
            self.wnLayerFavg = LinearDiag(nFeat)
            self.wnLayerWatt = LinearDiag(nFeat)
        else:
            raise ValueError('Not supported/recognized type {0}'.format(
                self.weight_generator_type))


    def get_classification_weights(
            self, Kbase_ids, features_train=None, labels_train=None):
        """Gets the classification weights of the base and novel categories.

        This routine returns the classification weight of the base categories
        and also (if training data, i.e., features_train and labels_train, for
        the novel categories are provided) of the novel categories.

        Args:
            Kbase_ids: A 2D tensor with shape [batch_size x nKbase] that for
                each training episode in the the batch it includes the indices
                of the base categories that are being used. `batch_size` is the
                number of training episodes in the batch and `nKbase` is the
                number of base categories.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the features of the training examples of each training episode
                in the batch. `num_train_examples` is the number of train
                examples in each training episode. Those training examples are
                from the novel categories.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the labels (encoded as 1-hot vectors of lenght nKnovel) of the
                training examples of each training episode in the batch.
                `nKnovel` is the number of novel categories.

        Returns:
            cls_weights: A 3D tensor of shape [batch_size x nK x num_channels]
                that includes the classification weight vectors
                (of `num_channels` length) of categories involved on each
                training episode in the batch. If training data for the novel
                categories are provided (i.e., features_train or labels_train
                are None) then cls_weights includes only the classification
                weights of the base categories; in this case nK is equal to
                nKbase. Otherwise, cls_weights includes the classification
                weights of both base and novel categories; in this case nK is
                equal to nKbase + nKnovel.
        """

        #***********************************************************************
        #******** Get the classification weights for the base categories *******
        batch_size, nKbase = Kbase_ids.size()
        weight_base = self.weight_base[Kbase_ids.view(-1)]
        weight_base = weight_base.view(batch_size, nKbase, -1)
        #***********************************************************************

        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        #***********************************************************************
        #******* Generate classification weights for the novel categories ******
        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)
        if self.classifier_type=='cosine':
            features_train = F.normalize(
                features_train, p=2, dim=features_train.dim()-1, eps=1e-12)
        if self.weight_generator_type=='none':
            weight_novel = self.favgblock(features_train, labels_train)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='feature_averaging':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='attention_based':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel_avg = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            if self.classifier_type=='cosine':
                weight_base_tmp = F.normalize(
                    weight_base, p=2, dim=weight_base.dim()-1, eps=1e-12)
            else:
                weight_base_tmp = weight_base

            weight_novel_att = self.attblock(
                features_train, labels_train, weight_base_tmp, Kbase_ids)
            weight_novel_att = self.wnLayerWatt(
                weight_novel_att.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel_avg + weight_novel_att
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        else:
            raise ValueError('Not supported / recognized type {0}'.format(
                self.weight_generator_type))
        #***********************************************************************

        # Concatenate the base and novel classification weights and return them.
        weight_both = torch.cat([weight_base, weight_novel], dim=1)
        # weight_both shape: [batch_size x (nKbase + nKnovel) x num_channels]

        return weight_both


    def apply_classification_weights(self, features, cls_weights):
        """Applies the classification weight vectors to the feature vectors.

        Args:
            features: A 3D tensor of shape
                [batch_size x num_test_examples x num_channels] with the feature
                vectors (of `num_channels` length) of each example on each
                trainining episode in the batch. `batch_size` is the number of
                training episodes in the batch and `num_test_examples` is the
                number of test examples of each training episode.
            cls_weights: A 3D tensor of shape [batch_size x nK x num_channels]
                that includes the classification weight vectors
                (of `num_channels` length) of the `nK` categories used on
                each training episode in the batch. `nK` is the number of
                categories (e.g., the number of base categories plus the number
                of novel categories) used on each training episode.

        Return:
            cls_scores: A 3D tensor with shape
                [batch_size x num_test_examples x nK] that represents the
                classification scores of the test examples for the `nK`
                categories.
        """
        if self.classifier_type=='cosine':
            features = F.normalize(
                features, p=2, dim=features.dim()-1, eps=1e-12)
            cls_weights = F.normalize(
                cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0,
            self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
        return cls_scores


    def forward(self, features_test, Kbase_ids, features_train=None, labels_train=None):
        """Recognize on the test examples both base and novel categories.

        Recognize on the test examples (i.e., `features_test`) both base and
        novel categories using the approach proposed on our CVPR2018 paper
        "Dynamic Few-Shot Visual Learning without Forgetting". In order to
        classify the test examples the provided training data for the novel
        categories (i.e., `features_train` and `labels_train`) are used in order
        to generate classification weight vectors of those novel categories and
        then those classification weight vectors are applied on the features of
        the test examples.

        Args:
            features_test: A 3D tensor with shape
                [batch_size x num_test_examples x num_channels] that represents
                the features of the test examples each training episode in the
                batch. Those examples can come both from base and novel
                categories. `batch_size` is the number of training episodes in
                the batch, `num_test_examples` is the number of test examples
                in each training episode, and `num_channels` is the number of
                feature channels.
            Kbase_ids: A 2D tensor with shape [batch_size x nKbase] that for
                each training episode in the the batch it includes the indices
                of the base categories that are being used. `nKbase` is the
                number of base categories.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the features of the training examples of each training episode
                 in the batch. `num_train_examples` is the number of train
                examples in each training episode. Those training examples are
                from the novel categories. If features_train is None then the
                current function will only return the classification scores for
                the base categories.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the labels (encoded as 1-hot vectors of lenght nKnovel) of the
                training examples of each training episode in the batch.
                `nKnovel` is the number of novel categories. If labels_train is
                None then the current function will return only the
                classification scores for the base categories.

        Return:
            cls_scores: A 3D tensor with shape
                [batch_size x num_test_examples x (nKbase + nKnovel)] that
                represents the classification scores of the test examples
                for the nKbase and nKnovel novel categories. If features_train
                or labels_train are None the only the classification scores of
                the base categories are returned. In that case the shape of
                cls_scores is [batch_size x num_test_examples x nKbase].
        """
        cls_weights = self.get_classification_weights(
            Kbase_ids, features_train, labels_train)
        cls_scores = self.apply_classification_weights(
            features_test, cls_weights)
        return cls_scores


def create_model(opt):
    return Classifier(opt)
