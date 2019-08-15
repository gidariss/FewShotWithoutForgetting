# *Dynamic Few-Shot Visual Learning without Forgetting*

### Introduction

The current project page provides [pytorch](http://pytorch.org/) code that implements the following CVPR2018 paper:   
**Title:**      "Dynamic Few-Shot Visual Learning without Forgetting"    
**Authors:**     Spyros Gidaris, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech      
**Code:**        https://github.com/gidariss/FewShotWithoutForgetting    
**Arxiv:**       https://arxiv.org/abs/1804.09458     

**Abstract:**  
The human visual system has the remarkably ability to be able to effortlessly learn novel concepts from only a few examples. Mimicking the same behavior on machine learning vision systems is an interesting and very challenging research problem with many practical advantages on real world vision applications. In this context, the goal of our work is to devise a few-shot visual learning system that during test time it will be able to efficiently learn novel categories from only a few training data while at the same time it will not forget the initial categories on which it was trained (here called base categories). To achieve that goal we propose (a) to extend an object recognition system with an attention based few-shot classification weight generator, and (b) to redesign the classifier of a ConvNet model as the cosine similarity function between feature representations and classification weight vectors. The latter, apart from unifying the recognition of both novel and base categories, it also leads to feature representations that generalize better on unseen categories. We extensively evaluate our approach on Mini-ImageNet where we manage to improve the prior state-of-the-art on few-shot recognition (i.e., we achieve $56.20\%$ and $73.00\%$ on the 1-shot and 5-shot settings respectively) while at the same time we do not sacrifice any accuracy on the base categories, which is a characteristic that most prior approaches lack. Finally, we apply our approach on the recently introduced few-shot benchmark of Bharath and Girshick where we also achieve state-of-the-art results.

### Citing FewShotWithoutForgetting

If you find the code useful in your research, please consider citing our CVPR2018 paper:
```
@inproceedings{gidaris2018dynamic,
  title={Dynamic Few-Shot Visual Learning without Forgetting},
  author={Gidaris, Spyros and Komodakis, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4367--4375},
  year={2018}
}
```

### Requirements
It was developed and tested with pytorch version 0.2.0_4

### License
This code is released under the MIT License (refer to the LICENSE file for details). 

## Running experiments on MiniImageNet.

First, you must download the MiniImagenet dataset from [here](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) and set in [dataloader.py](https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py#L28) the path to where the dataset resides in your machine. We recommend creating a *dataset* directory `mkdir datasets` and placing the downloaded dataset there.

### Training and evaluating our model on Mini-ImageNet.

**(1)** In order to run the 1st training stage of our approach (which trains a recognition model with a cosine-similarity based classifier and a feature extractor with 128 feature channels on its last convolution layer) run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
```
The above command launches the training routine using the configuration file `./config/miniImageNet_Conv128CosineClassifier.py` which is specified by the `--config` argument (i.e., `--config=miniImageNet_Conv128CosineClassifier`). Note that all the experiment configuration files are placed in the [./config](https://github.com/gidariss/FewShotWithoutForgetting/tree/master/config) directory.

**(2)** In order to run the 2nd training state of our approach (which trains the few-shot classification weight generator with attenition based weight inference) run the following commands:
```
# Training the model for the 1-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1
# Training the model for the 5-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 
```

**(3)** In order to evaluate the above models run the following commands:
```
# Evaluating the model for the 1-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 --testset
# Evaluating the model for the 5-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 --testset
```

**(4)** In order to train and evaluate our approach with different type of feature extractors (e.g., Conv32, Conv64, or ResNetLike; see our paper for a desciption of those feature extractors) run the following commands:
```
#************************** Feature extractor: Conv32 *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifier
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN5 --testset

#************************** Feature extractor: Conv64 *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifier
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN5 --testset

#************************** Feature extractor: ResNetLike *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosine
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5 --testset
```

### Training and evaluating Matching Networks or Prototypical Networks on Mini-ImageNet.

In order to train and evaluate our implementations of Matching Networks[3] and Prototypical Networks[4] run the following commands:
```
# Train and evaluate the matching networks model for the 1-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128MatchingNetworkN1 --testset

# Train and evaluate the matching networks model for the 5-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128MatchingNetworkN5 --testset

# Train and evaluate the prototypical networks model for the 1-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128PrototypicalNetworkN1 --testset

# Train and evaluate the prototypical networks model for the 5-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128PrototypicalNetworkN5 --testset
```

### Experimental results on the test set of Mini-ImageNet.

Here we provide experimental results of our approach as well as of our implementations of Matching Networks and Prototypical Networks on the test of a Mini-ImageNet. Note that after cleaning and refactoring the implementation code of the paper and re-running the experiments, the results that we got are slightly different. 

#### 1-shot 5-way classification accuracy of novel categories

| Approach                                   | Feature extractor | Novel           | Base   | Both   |
|--------------------------------------------| ----------------- | ---------------:| ------:| ------:|
| Matching Networks [3]                      | Conv64            | 43.60%          | -      | -      |
| Prototypical Networks [4]                  | Conv64            | 49.42% +/- 0.78 | -      | -      |
| Ravi and Larochelle [5]                    | Conv32            | 43.40% +/- 0.77 | -      | -      |
| Finn et al [6]                             | Conv64            | 48.70% +/- 1.84 | -      | -      |
| Mishra et al [6]                           | ResNet            | 55.71% +/- 0.99 | -      | -      |
| Matching Networks (our implementation)     | Conv64            | 53.65% +/- 0.80 | -      | -      |
| Matching Networks (our implementation)     | Conv128           | 54.32% +/- 0.80 | -      | -      |
| Prototypical Networks (our implementation) | Conv64            | 53.30% +/- 0.79 | -      | -      |
| Prototypical Networks (our implementation) | Conv128           | 54.14% +/- 0.80 | -      | -      |
| Prototypical Networks (our implementation) | ResNet            | 53.74% +/- 0.91 | -      | -      |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv32            | 54.49% +/- 0.83 | 61.59% | 44.79% |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv64            | 55.86% +/- 0.85 | 68.43% | 47.75% |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv128           | **56.62% +/- 0.84** | 70.80% | 49.44% |
| **(Ours)** Cosine & Att. Weight Gen.       | ResNet            | 56.21% +/- 0.83 | **79.79%** | **52.81%** |


#### 5-shot 5-way classification accuracy of novel categories

| Approach                                   | Feature extractor | Novel           | Base   | Both    |
| ------------------------------------------ | ----------------- |----------------:| ------:|--------:|
| Matching Networks [3]                      | Conv64            | 55.30%          | -      | -       |
| Prototypical Networks [4]                  | Conv64            | 68.20% +/- 0.66 | -      | -       |
| Ravi and Larochelle [5]                    | Conv32            | 60.20% +/- 0.71 | -      | -       |
| Finn et al [6]                             | Conv64            | 63.10% +/- 0.92 | -      | -       |
| Mishra et al [6]                           | ResNet            | 68.88% +/- 0.92 | -      | -       |
| Matching Networks (our implementation)     | Conv64            | 65.76% +/- 0.68 | -      | -       |
| Matching Networks (our implementation)     | Conv128           | 65.97% +/- 0.65 | -      | -       |
| Prototypical Networks (our implementation) | Conv64            | 70.33% +/- 0.65 | -      | -       |
| Prototypical Networks (our implementation) | Conv128           | 70.74% +/- 0.66 | -      | -       |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv32            | 70.12% +/- 0.67 | 60.83% | 53.22%  |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv64            | 72.49% +/- 0.62 | 67.64% | 57.21%  |
| **(Ours)** Cosine & Att. Weight Gen.       | Conv128           | **72.82% +/- 0.63** | 71.00% | 59.05%  |
| **(Ours)** Cosine & Att. Weight Gen.       | ResNet            | 70.64% +/- 0.66 | **79.56%** | **59.48%**  |


## Running experiments on the ImageNet based Low-shot benchmark

Here provide instructions on how to train and evaluate our approach on the ImageNet based low-shot benchmark proposed by Bharath and Girshick [1]. 

**(1)** First, you must download the ImageNet dataset and set in [dataloader.py](https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py#L29) the path to where the dataset resides in your machine. We recommend creating a *dataset* directory `mkdir datasets` and placing the downloaded dataset there. 

**(2)** Launch the 1st training stage of our approach by running the following command:
```
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage1.py --config=imagenet_ResNet10CosineClassifier
```
The above command will train the a recognition model with a ResNet10 feature extractor and a cosine similarity based classifier for 100 epochs (which will take around ~120 hours). You can download the already trained by us recognition model from [here](https://mega.nz/#!fw12RApC!RCnaQd-iEdQuMVZYBFAcPOJKxqrV1Q0m1uTGw6xwDio). In that case you should place the model inside the './experiments' directory with the name './experiments/imagenet_ResNet10CosineClassifier'.

**(3)** Extract and save the ResNet10 features (with the model that we trained above) from images of the ImageNet dataset:
```
# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=train
```

**(4)** Launch the 2st training stage of our approach (which trains the few-shot classification weight generator with attenition based weight inference) by running the following commands:
```
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
```

**(5)** Evaluate the above trained models by running the following commands:
```
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
```

## Experimental results on the ImageNet based Low-shot benchmark

Here we evaluate our approach on the ImageNet based low-shot benchmark proposed by Bharath and Girshick [1] using the improved evaluation metrics proposed by Wang et al [2]. All the approaches use a ResNet10 feature extractor. Note that after cleaning and refactoring the implementation code of the paper and re-running the experiments, the results that we got are slightly different. 
A pre-trained ResNet10 model with cosine-similarity based classifier is provided here: [imagenet_ResNet10CosineClassifier](https://mega.nz/#!fw12RApC!RCnaQd-iEdQuMVZYBFAcPOJKxqrV1Q0m1uTGw6xwDio) 


### Top-5 1-shot classification accuracy
 
| Approach                             | Novel           | All             | All with prior  | 
| ------------------------------------ | ---------------:|----------------:| ---------------:|
| Prototypical Networks (from[2])      | 39.30%          | 49.50%          | 53.60%          |
| Matching Networks (from[2])          | 43.60%          | 54.40%          | 54.50%          |
| Logistic regression (from[2])        | 38.40%          | 40.80%          | 52.90%          |
| Logistic regression w/ H (from[2])   | 40.70%          | 52.20%          | 53.20%          |
| Prototype Matching Nets [2]          | 43.30%          | 55.80%          | 54.70%          |
| Prototype Matching Nets w/ H [2]     | 45.80%          | 57.60%          | 56.40%          |
| **(Ours)** Cosine & Att. Weight Gen. | **46.26% +/- 0.20** | **58.29% +/- 0.13** | **56.88% +/- 0.13** |

### Top-5 2-shot classification accuracy

| Approach                             | Novel           | All             | All with prior  | 
| ------------------------------------ | ---------------:|----------------:| ---------------:|
| Prototypical Networks (from[2])      | 54.40%          | 61.00%          | 61.40%          |
| Matching Networks (from[2])          | 54.00%          | 61.00%          | 60.70%          |
| Logistic regression (from[2])        | 51.10%          | 49.90%          | 60.40%          |
| Logistic regression w/ H (from[2])   | 50.80%          | 59.40%          | 59.10%          |
| Prototype Matching Nets [2]          | 55.70%          | 63.10%          | 62.00%          |
| Prototype Matching Nets w/ H [2]     | **57.80%**      | 64.70%          | 63.30%          |
| **(Ours)** Cosine & Att. Weight Gen. | 57.46% +/- 0.16 | **65.11% +/- 0.10** | **63.67% +/- 0.09** |

### Top-5 5-shot classification accuracy

| Approach                             | Novel           | All             | All with prior  | 
| ------------------------------------ | ---------------:|----------------:| ---------------:|
| Prototypical Networks (from[2])      | 66.30%          | 69.70%          | 68.80%          |
| Matching Networks (from[2])          | 66.00%          | 69.00%          | 68.20%          |
| Logistic regression (from[2])        | 64.80%          | 64.20%          | 68.60%          |
| Logistic regression w/ H (from[2])   | 62.00%          | 67.60%          | 66.80%          |
| Prototype Matching Nets [2]          | 68.40%          | 71.10%          | 70.20%          |
| Prototype Matching Nets w/ H [2]     | 69.00%          | 71.90%          | 70.60%          |
| **(Ours)** Cosine & Att. Weight Gen. | **69.27% +/- 0.09** | **72.70% +/- 0.06** | **71.24% +/- 0.06** |

### Top-5 10-shot classification accuracy

| Approach                             | Novel           | All             | All with prior  | 
| ------------------------------------ | ---------------:|----------------:| ---------------:|
| Prototypical Networks (from[2])      | 71.20%          | 72.90%          | 72.00%          |
| Matching Networks (from[2])          | 72.50%          | 73.70%          | 72.60%          |
| Logistic regression (from[2])        | 71.60%          | 71.90%          | 72.90%          |
| Logistic regression w/ H (from[2])   | 69.30%          | 72.80%          | 71.70%          |
| Prototype Matching Nets [2]          | 74.00%          | 75.00%          | 73.90%          |
| Prototype Matching Nets w/ H [2]     | 74.30%          | 75.20%          | 74.00%          | 
| **(Ours)** Cosine & Att. Weight Gen. | **74.84% +/- 0.06** | **76.51% +/- 0.04** | **75.00% +/- 0.04** |

### Top-5 20-shot classification accuracy

| Approach                             | Novel           | All             | All with prior  | 
| ------------------------------------ | ---------------:|----------------:| ---------------:|
| Prototypical Networks (from[2])      | 73.90%          | 74.60%          | 73.80%          |
| Matching Networks (from[2])          | 76.90%          | 76.50%          | 75.60%          |
| Logistic regression (from[2])        | 76.60%          | 76.90%          | 76.30%          |
| Logistic regression w/ H (from[2])   | 76.50%          | 76.90%          | 76.30%          |
| Prototype Matching Nets [2]          | 77.00%          | 77.10%          | 75.90%          |
| Prototype Matching Nets w/ H [2]     | 77.40%          | 77.50%          | 76.20%          |
| **(Ours)** Cosine & Att. Weight Gen. | **78.11% +/- 0.05** | **78.74% +/- 0.03** | **77.28% +/- 0.03** |

### References
```
[1] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[2] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
[3] O. Vinyals et al. Matching networks for one shot learning.
[4] J. Snell, K. Swersky, and R. S. Zemel. Prototypical networks for few-shot learning.
[5] S. Ravi and H. Larochelle. Optimization as a model for few-shot learning.
[6] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks.
```
