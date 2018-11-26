# A Simple Cache Model for Image Recognition

This repository contains the code for reproducing the results reported in the following paper: 

Orhan AE (2018) [A simple cache model for image recognition.](https://arxiv.org/abs/1805.08709) *NIPS 2018* [arxiv:1805.08709].

The code was written and tested in Tensorflow (v1.4.0) and Keras (v2.0.9) on a GPU cluster. Other configurations may or may not work. Please let me know if you have any trouble running the code. A brief description of the directories follows:

+ `adversarial`: contains code for running both black-box and white-box attacks against the baseline and cache models. The code here requires the [Foolbox](https://github.com/bethgelab/foolbox) toolbox.
+ `depth-expts`: contains code for testing the effect of layer depth in cache models, as reported in Fig. 2 in the paper. 
+ `imagenet`: contains code for running ImageNet experiments. I had to divide the training and validation data into several chunks to deal with memory issues (see the pre-processing files in the folder). This may or may not be suitable for your needs.
+ `saved_models`: contains saved ResNet and DenseNet models trained on the CIFAR-10 and CIFAR-100 datasets.  

Files with a `_grid` in their name can be used to run hyper-parameter searches. Files with a `_evaluate` in their name can be used to evaluate the models. Files with a `_justmem` in their name are associated with the cache-only (CacheOnly) models.
