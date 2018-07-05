from __future__ import print_function

import numpy as np
from dataloader import MiniImageNet, ImageNetLowShot, FewShotDataloader

from pdb import set_trace as breakpoint


def test_sample_episode(dataset):
    dloader = FewShotDataloader(
        dataset=dataset,
        nKnovel=5,
        nKbase=-1,
        nExemplars=1,
        nTestNovel=15*2,
        nTestBase=15*2,
        batch_size=1,
        num_workers=1,
        epoch_size=2)
    nKnovel = dloader.nKnovel
    nKbase = dloader.nKbase
    nTestBase = dloader.nTestBase
    nTestNovel = dloader.nTestNovel
    nExemplars = dloader.nExemplars

    label2ind = dloader.dataset.label2ind
    labelIds_base = dloader.dataset.labelIds_base
    all_img_ids_of_base_categories = []
    for label_id in labelIds_base:
        all_img_ids_of_base_categories += label2ind[label_id]

    if dloader.phase != 'train':
        labelIds_novel = dloader.dataset.labelIds_novel
        all_img_ids_of_novel_categories = []
        for label_id in labelIds_novel:
            all_img_ids_of_novel_categories += label2ind[label_id]

    for _ in range(100):
        Exemplars, Test, Kall, nKbase = dloader.sample_episode()

        Knovel = Kall[nKbase:] # Category ids of the base categories.
    	# Verify that the image ids of the training exemples for the novel
        # categories do not overal with the test examples for the novel
        # categories.
        test_novel = list(filter(lambda (_, label): (label >= nKbase), Test))
        test_novel = [img_id for (img_id, _ ) in test_novel]
        train_novel = [img_id for (img_id, _ ) in Exemplars]
        assert(len(set.intersection(set(test_novel),set(train_novel))) == 0)

        if dloader.phase != 'train':
            # Verify that the image id that are coming from the novel categories
            # (from both the training and the test examples of the novel
            # categoriesdo not belong training categories (i.e., the categories
            # that are used for training the model during the training
            # procedure).
            novel_img_ids = test_novel + train_novel

            assert(
                len(set.intersection(
                    set(all_img_ids_of_base_categories),
                    set(novel_img_ids))) == 0
            )
            assert(
                len(set.intersection(
                    set(all_img_ids_of_novel_categories),
                    set(novel_img_ids))) == len(novel_img_ids)
            )

        # Verify Exemplars list.
        histE = [0 for i in range(nKnovel+nKbase)]
        for (_, label) in Exemplars: histE[label] += 1
        # Test that the label ids in the examplars list do not belong on the
        # base categories.
        assert(all(val==0 for val in histE[:nKbase]))
        # Test that the label ids in the examplars list belong on the novel
        # categories and that for each novel category 'nExemplars' number of
        # examples have been sampled.
        assert(all(val==nExemplars for val in histE[nKbase:]))

        # Verify Test list.
        histT = [0 for i in range(nKnovel+nKbase)]
        for (_, label) in Test: histT[label] += 1
        # Test that the number of test examples comming from the base categories
        # is equal to nTestBase.
        if nKbase != 0:
            assert(reduce(lambda x,y: x+y, histT[:nKbase]) == nTestBase)
        # Test that the number of test examples comming from the novel
        # categories is equal to nTestNovel.
        if nKnovel != 0:
            assert(reduce(lambda x,y: x+y, histT[nKbase:]) == nTestNovel)

        # Verify that the Kbase and Knovel categories do not intersect.
        assert(
            len(set.intersection(set(Kall[:nKbase]),set(Kall[nKbase:]))) == 0
        )


if __name__ == '__main__':
    test_sample_episode(MiniImageNet(phase='train'))
    print("The tests for the training phase of the dataloader were passed.")
    test_sample_episode(MiniImageNet(phase='val'))
    print("The tests for the validation phase of the dataloader were passed.")
    test_sample_episode(MiniImageNet(phase='test'))
    print("The tests for the testing phase of the dataloader were passed.")

    test_sample_episode(ImageNetLowShot(phase='train'))
    print("The tests for the training phase of the dataloader were passed.")
    test_sample_episode(ImageNetLowShot(phase='val'))
    print("The tests for the validation phase of the dataloader were passed.")
    test_sample_episode(ImageNetLowShot(phase='test'))
    print("The tests for the testing phase of the dataloader were passed.")
