
from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt

import h5py

from PIL import Image
from PIL import ImageEnhance

from pdb import set_trace as breakpoint


# Set the appropriate paths of the datasets here.
_MINI_IMAGENET_DATASET_DIR = './datasets/MiniImagenet'
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'

_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH = './data/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json'

transformtypedict=dict(
    Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color
)


def get_label_ids(class_to_idx, class_names, inside=True):
    label_ids = []
    if inside:
        for cname in class_names:
            label_ids.append(class_to_idx[cname])
    else:
        for cname, clabel in class_to_idx.items():
            if cname not in class_names:
                label_ids.append(clabel)

    return label_ids


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


class MiniImageNet(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        self.base_folder = 'miniImagenet'
        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'MiniImageNet_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_train.pickle')
        file_train_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_val.pickle')
        file_train_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_test.pickle')
        file_val_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_test.pickle')

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate(
                [data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = nTestNovel / nKnovel

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return (self.epoch_size / self.batch_size)


class ImageNetLowShot(data.Dataset):
    def __init__(self,
                 phase='train',
                 split='train',
                 do_not_use_random_transf=False):
        self.phase = phase
        self.split = split
        assert(phase=='train' or phase=='test' or phase=='val')
        assert(split=='train' or split=='val')
        self.name = 'ImageNetLowShot_Phase_' + phase + '_Split_' + split

        print('Loading ImageNet dataset (for few-shot benchmark) - phase {0}'.
            format(phase))

        #***********************************************************************
        with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
            label_idx = json.load(f)
        base_classes = label_idx['base_classes']
        novel_classes_val_phase = label_idx['novel_classes_1']
        novel_classes_test_phase = label_idx['novel_classes_2']
        #***********************************************************************

        transforms_list = []
        if (phase!='train') or (do_not_use_random_transf==True):
            transforms_list.append(transforms.Scale(256))
            transforms_list.append(transforms.CenterCrop(224))
        else:
            transforms_list.append(transforms.RandomSizedCrop(224))
            jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}
            transforms_list.append(ImageJitter(jitter_params))
            transforms_list.append(transforms.RandomHorizontalFlip())

        transforms_list.append(lambda x: np.asarray(x))
        transforms_list.append(transforms.ToTensor())
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        transforms_list.append(transforms.Normalize(mean=mean_pix, std=std_pix))

        self.transform = transforms.Compose(transforms_list)

        traindir = os.path.join(_IMAGENET_DATASET_DIR, 'train')
        valdir = os.path.join(_IMAGENET_DATASET_DIR, 'val')
        self.data = datasets.ImageFolder(
            traindir if split=='train' else valdir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert(self.num_cats==1000)

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val_phase if (self.phase=='val') else
                novel_classes_test_phase)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNet(data.Dataset):
    def __init__(self, split='train'):
        self.split = split
        assert(split=='train' or split=='val')
        self.name = 'ImageNet_Split_' + split

        print('Loading ImageNet dataset - split {0}'.format(split))
        transforms_list = []
        transforms_list.append(transforms.Scale(256))
        transforms_list.append(transforms.CenterCrop(224))
        transforms_list.append(lambda x: np.asarray(x))
        transforms_list.append(transforms.ToTensor())
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        transforms_list.append(transforms.Normalize(mean=mean_pix, std=std_pix))
        self.transform = transforms.Compose(transforms_list)

        traindir = os.path.join(_IMAGENET_DATASET_DIR, 'train')
        valdir = os.path.join(_IMAGENET_DATASET_DIR, 'val')
        self.data = datasets.ImageFolder(
            traindir if split=='train' else valdir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class SimpleDataloader():
    def __init__(self, dataset, batch_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = len(dataset)

    def get_iterator(self):
        def load_fun_(idx):
            img, label = self.dataset[idx]
            return img, label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_fun_)

        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, drop_last=False)

        return data_loader

    def __call__(self):
        return self.get_iterator()

    def __len__(self):
        return (self.epoch_size / self.batch_size)


class ImageNetLowShotFeaturesLegacy():
    def __init__(
        self,
        data_dir,
        phase='train',
        add_novel_split='val'):

        self.phase = phase
        assert(phase=='train' or phase=='test' or phase=='val')
        self.name = 'ImageNetLowShotFeatures_Phase_' + phase

        split = 'train' if (phase=='train') else 'val'
        dataset_file = os.path.join(data_dir, 'feature_dataset_'+split+'.json')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        #***********************************************************************
        with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
            label_idx = json.load(f)
        base_classes = label_idx['base_classes']
        base_classes_val_phase = label_idx['base_classes_1']
        base_classes_test_phase = label_idx['base_classes_2']
        novel_classes_val_phase = label_idx['novel_classes_1']
        novel_classes_test_phase = label_idx['novel_classes_2']
        #***********************************************************************

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert(self.num_cats==1000)

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)

        novel_split = add_novel_split if (self.phase=='train') else self.phase
        if novel_split=='val' or novel_split=='test':
            self.labelIds_novel = (
                novel_classes_val_phase if (novel_split=='val') else
                novel_classes_test_phase)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)

            self.base_classes_subset = (
                base_classes_val_phase if (novel_split=='val') else
                base_classes_test_phase)
        else:
            self.base_classes_subset = base_classes

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)


class ImageNetLowShotFeatures():
    def __init__(
        self,
        data_dir, # path to the directory with the saved ImageNet features.
        image_split='train', # the image split of the ImageNet that will be loaded.
        phase='train', # whether the dataset will be used for training, validating, or testing a model.
        ):
        assert(image_split=='train' or image_split=='val')
        assert(phase=='train' or phase=='val' or phase=='test')

        self.phase = phase
        self.image_split = image_split
        self.name = ('ImageNetLowShotFeatures_ImageSplit_' + self.image_split
                     +'_Phase_' + self.phase)

        dataset_file = os.path.join(
            data_dir, 'feature_dataset_' + self.image_split + '.json')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        #***********************************************************************
        with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
            label_idx = json.load(f)
        base_classes = label_idx['base_classes']
        base_classes_val_split = label_idx['base_classes_1']
        base_classes_test_split = label_idx['base_classes_2']
        novel_classes_val_split = label_idx['novel_classes_1']
        novel_classes_test_split = label_idx['novel_classes_2']
        #***********************************************************************

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert(self.num_cats==1000)

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)

        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val_split if (self.phase=='val') else
                novel_classes_test_split)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
            self.base_classes_eval_split = (
                base_classes_val_split if (self.phase=='val') else
                base_classes_test_split)
            self.base_classes_subset = self.base_classes_eval_split

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)


class LowShotDataloader():
    def __init__(
        self,
        dataset_train_novel,
        dataset_evaluation,
        nExemplars=1,
        batch_size=1,
        num_workers=4):

        self.nExemplars = nExemplars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_train_novel = dataset_train_novel
        self.dataset_evaluation = dataset_evaluation

        assert(self.dataset_evaluation.labelIds_novel ==
               self.dataset_train_novel.labelIds_novel)

        assert(self.dataset_evaluation.labelIds_base ==
               self.dataset_train_novel.labelIds_base)

        assert(self.dataset_evaluation.base_classes_eval_split ==
               self.dataset_train_novel.base_classes_eval_split)

        self.nKnovel = self.dataset_evaluation.num_cats_novel
        self.nKbase = self.dataset_evaluation.num_cats_base

        # Category ids of the base categories.
        self.Kbase = sorted(self.dataset_evaluation.labelIds_base)
        assert(self.nKbase == len(self.Kbase))
        # Category ids of the novel categories.
        self.Knovel = sorted(self.dataset_evaluation.labelIds_novel)
        assert(self.nKnovel == len(self.Knovel))

        self.Kall = self.Kbase + self.Knovel

        self.CategoryId2LabelIndex = {
            category_id: label_index for label_index, category_id in enumerate(self.Kall)
        }
        self.Kbase_eval_split = self.dataset_train_novel.base_classes_eval_split

        Kbase_set = set(self.Kall[:self.nKbase])
        Kbase_eval_split_set = set(self.Kbase_eval_split)
        assert(len(set.intersection(Kbase_set, Kbase_eval_split_set)) == len(Kbase_eval_split_set))

        self.base_eval_split_labels = sorted(
            [self.CategoryId2LabelIndex[category_id] for category_id in self.Kbase_eval_split]
        )

        # Collect the image indices of the evaluation set for both the base and
        # the novel categories.
        data_indices = []
        for category_id in self.Kbase_eval_split:
            data_indices += self.dataset_evaluation.label2ind[category_id]
        for category_id in self.Knovel:
            data_indices += self.dataset_evaluation.label2ind[category_id]
        self.eval_data_indices = sorted(data_indices)
        self.epoch_size = len(self.eval_data_indices)

    def base_category_label_indices(self):
        return self.base_eval_split_labels

    def novel_category_label_indices(self):
        return range(self.nKbase, len(self.Kall))

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset_train_novel.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset_train_novel.label2ind)
        assert(len(self.dataset_train_novel.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset_train_novel.label2ind[cat_id], sample_size)

    def sample_training_examples_for_novel_categories(
        self, Knovel, nExemplars, nKbase):
        """Samples (a few) training examples for the novel categories.

        Args:
            Knovel: a list with the ids of the novel categories.
            nExemplars: the number of training examples per novel category.
            nKbase: the number of base categories.

        Returns:
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """
        Exemplars = []
        for knovel_idx, knovel_label in enumerate(Knovel):
            imds = self.sampleImageIdsFrom(knovel_label, sample_size=nExemplars)
            Exemplars += [(img_id, nKbase + knovel_idx) for img_id in imds]
        random.shuffle(Exemplars)

        return  Exemplars

    def create_examples_tensor_data(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset_train_novel[img_idx][0] for img_idx, _ in examples],
            dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def sample_training_data_for_novel_categories(self, exp_id=0):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        random.seed(exp_id) # fix the seed for this experiment.
        # Sample `nExemplars` number of training examples per novel category.
        train_examples = self.sample_training_examples_for_novel_categories(
            self.Knovel, self.nExemplars, nKbase)
        Kall = torch.LongTensor(self.Kall)
        images_train, labels_train = self.create_examples_tensor_data(
            train_examples)

        return images_train, labels_train, Kall, nKbase, nKnovel

    def get_iterator(self, epoch=0):
        def load_fun_(idx):
            img_idx = self.eval_data_indices[idx]
            img, category_id = self.dataset_evaluation[img_idx]
            label = (self.CategoryId2LabelIndex[category_id]
                     if (category_id in self.CategoryId2LabelIndex) else -1)
            return img, label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_fun_)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=False)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(math.ceil(float(self.epoch_size)/self.batch_size))


class LowShotDataloaderLegacy():
    def __init__(
        self,
        dataset_train_novel,
        dataset_evaluation,
        nExemplars=1,
        batch_size=1,
        num_workers=4):

        self.dataset_train_novel  = dataset_train_novel
        self.dataset_evaluation = dataset_evaluation

        assert(self.dataset_evaluation.labelIds_novel ==
               self.dataset_train_novel.labelIds_novel)

        # Collect the image indices of the evaluation set for both the base and
        # the novel categories.
        data_inds = []
        for kid in self.dataset_evaluation.labelIds_base:
            data_inds += self.dataset_evaluation.label2ind[kid]
        for kid in self.dataset_evaluation.labelIds_novel:
            data_inds += self.dataset_evaluation.label2ind[kid]
        self.eval_data_indices = sorted(data_inds)

        self.nKnovel = self.dataset_evaluation.num_cats_novel
        self.nKbase = self.dataset_evaluation.num_cats_base

        self.nExemplars = nExemplars
        self.batch_size = batch_size
        self.epoch_size = len(self.eval_data_indices)
        self.num_workers = num_workers

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset_train_novel.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset_train_novel.label2ind)
        assert(len(self.dataset_train_novel.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset_train_novel.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset_evaluation.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset_evaluation.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_novel_data(self):
        """Samples a few training examples for each novel category."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nExemplars = self.nExemplars

        # Kbase = sorted(self.dataset_evaluation.labelIds_base)
        # Knovel = sorted(self.dataset_evaluation.labelIds_novel)
        Kbase = sorted(self.sampleCategories('base', nKbase))
        Knovel = sorted(self.sampleCategories('novel', nKnovel))

        Exemplars = []
        for knovel_idx, knovel_label in enumerate(Knovel):
            imds = self.sampleImageIdsFrom(knovel_label, sample_size=nExemplars)
            Exemplars += [(img_id, nKbase + knovel_idx) for img_id in imds]
        random.shuffle(Exemplars)

        Kids = Kbase + Knovel

        return  Exemplars, Kids, nKbase, nKnovel

    def create_examples_tensor_data(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset_train_novel[img_idx][0] for img_idx, _ in examples],
            dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def getNovelCategoriesTrainingData(self, exp_id=0):
        random.seed(exp_id) # fix the seed for this experiment.

        # Sample training examples for each novel category.
        Exemplars, Kids, nKbase, nKnovel = self.sample_novel_data()
        self.Kid2Label = {kid: label_idx for label_idx, kid in enumerate(Kids)}

        base_classes_subset = self.dataset_train_novel.base_classes_subset
        assert(len(set.intersection(set(Kids[:nKbase]),set(base_classes_subset))) == len(base_classes_subset))
        self.Kids_base_subset = sorted([self.Kid2Label[kid] for kid in base_classes_subset])

        Kids = torch.LongTensor(Kids)
        Xe, Ye = self.create_examples_tensor_data(Exemplars)
        return Xe, Ye, Kids, nKbase, nKnovel

    def sample_training_data_of_novel_categories(self, exp_id=0):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nExemplars = self.nExemplars

        random.seed(exp_id) # fix the seed for this experiment.
        breakpoint()
        # Ids of the base categories.
        Kbase = sorted(self.dataset_evaluation.labelIds_base)
        # Ids of the novel categories.
        Knovel = sorted(self.dataset_evaluation.labelIds_novel)
        assert(len(Kbase) == nKnovel and len(Knovel) == nKbase)
        Kall = Kbase + Knovel

        # Sample `nExemplars` number of training examples for each novel
        # category.
        train_examples = self.sample_training_examples_for_novel_categories(
            Knovel, nExemplars)

        breakpoint()
        self.Kid2Label = {kid: label_idx for label_idx, kid in enumerate(Kall)}

        breakpoint()
        base_classes_subset = self.dataset_train_novel.base_classes_subset
        assert(len(set.intersection(set(Kall[:nKbase]),set(base_classes_subset))) == len(base_classes_subset))
        self.Kids_base_subset = sorted([self.Kid2Label[kid] for kid in base_classes_subset])

        Kall = torch.LongTensor(Kall)
        images_train, labels_train = self.create_examples_tensor_data(train_examples)
        return images_train, labels_train, Kall, nKbase, nKnovel

    def get_iterator(self, epoch=0):
        def load_fun_(idx):
            img_idx = self.eval_data_indices[idx]
            img, kid = self.dataset_evaluation[img_idx]
            label = self.Kid2Label[kid] if (kid in self.Kid2Label) else -1
            return img, label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_fun_)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=False)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(math.ceil(float(self.epoch_size)/self.batch_size))
