resnet-tensorflow
=================

By [Lu Xu](https://github.com/LuXu1113).

Contents
--------
1. [Introduction](#introduction)
1. [Reference](#reference)
1. [Specifications](#specifications)
1. [Notice](#notice)
1. [Models](#models)
1. [Results](#results)

---

### Introduction

Written for learning Resnet and tensorflow. Based on python-3.6 and tensorflow-1.0.

### Reference

1. [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)
1. [deep-residual-networs](https://github.com/KaimingHe/deep-residual-networks)
1. [tensorflow-api-docs](http://www.tensorfly.cn/tfdoc/api_docs/index.html)

### Notice

1. tf.losses.sparse_softmax_cross_entropy will not report error when the rank of logits is equal the rank of labels plus 1, we should make sure of this, otherwise training will not work properly.
1. Following [API master](https://www.tensorflow.org/versions/master/api_docs/python/tf/layers/batch_normalization),when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op. For example:

        update_ops      = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops) :
            self.train  = optimizer().minimize(loss)

### Specifications

* In this implementation, projection shortcuts are used for increasing deminsions, and other shortcuts are identity, which is mentioned in the [paper](http://arxiv.org/abs/1512.03385) as option B.

### Results

* Train Resnet-20 on CIFAR:

[resnet-20 (CIFAR)] (https://github.com/LuXu1113/resnet-tensorflow/blob/master/models/cifar10_resnet20_train.png)

### Models

* Visualizations of network structures (draw by tensorboard)
    - [resnet-20 (CIFAR)](https://github.com/LuXu1113/resnet-tensorflow/blob/master/models/cifar10_resnet20.png)
    - [resnet-56 (CIFAR)]()
    - [resnet-110 (CIFAR)]()
    - [resnet-18 (ImageNet)]()
    - [resnet-50 (ImageNet)]()
    - [resnet-152 (ImageNet)]()
