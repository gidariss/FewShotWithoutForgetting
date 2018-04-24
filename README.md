## *Dynamic Few-Shot Visual Learning without Forgetting*

### Introduction

The current project page will provide [pytorch](http://pytorch.org/) that implements the following CVPR2018 paper:   
**Title:**      "Dynamic Few-Shot Visual Learning without Forgetting"    
**Authors:**     Spyros Gidaris, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech    
**Code:**        https://github.com/gidariss/FewShotWithoutForgetting   

**Abstract:**  
The human visual system has the remarkably ability to be able to effortlessly learn novel concepts from only a few examples. Mimicking the same behavior on machine learning vision systems is an interesting and very challenging research problem with many practical advantages on real world vision applications. In this context, the goal of our work is to devise a few-shot visual learning system that during test time it will be able to efficiently learn novel categories from only a few training data while at the same time it will not forget the initial categories on which it was trained (here called base categories). To achieve that goal we propose (a) to extend an object recognition system with an attention based few-shot classification weight generator, and (b) to redesign the classifier of a ConvNet model as the cosine similarity function between feature representations and classification weight vectors. The latter, apart from unifying the recognition of both novel and base categories, it also leads to feature representations that generalize better on unseen categories. We extensively evaluate our approach on Mini-ImageNet where we manage to improve the prior state-of-the-art on few-shot recognition (i.e., we achieve $56.20\%$ and $73.00\%$ on the 1-shot and 5-shot settings respectively) while at the same time we do not sacrifice any accuracy on the base categories, which is a characteristic that most prior approaches lack. Finally, we apply our approach on the recently introduced few-shot benchmark of Bharath and Girshick where we also achieve state-of-the-art results.

## The code will be soon available.
