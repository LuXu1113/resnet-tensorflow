# encoding: utf-8
# python-version: 3.6

import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

class ResNet :
    def __init__(self, data_set, depth) :
        self.is_training = tf.placeholder(tf.bool, [], name = "is_training")

        self.instances = None
        self.labels    = None
        logits = None

        if data_set == "CIFAR-10" :
            self.instances = tf.placeholder(tf.uint8, [None, 32, 32, 3], name = "instances")
            self.labels    = tf.placeholder(tf.uint8, [None, 1], name = "labels")

            if depth == 20 :
                logits = self.build_resnet_cifar(self.instances, self.is_training, 3)
            if depth == 56 :
                logits = self.build_resnet_cifar(self.instances, self.is_training, 9)
            if depth == 110 :
                logits = self.build_resnet_cifar(self.instances, self.is_training, 18)

        elif data_set == "ImageNet" :
            self.instances = tf.placeholder(tf.uint8, [None, 224, 224, 3], name = "instances")
            self.labels    = tf.placeholder(tf.uint8, [None, 1], name = "labels")

            if depth == 18 :
                logits = self.build_resnet_imagenet(self.instances, self.is_training, [2, 2, 2, 2])
            if depth == 50 :
                logits = self.build_resnet_imagenet(self.instances, self.is_training, [3, 4, 6, 3])
            if depth == 152 :
                logits = self.build_resnet_imagenet(self.instances, self.is_training, [3, 8, 36, 3])
               

        if None == logits:
            print("Resnet%d for dataset %s is not supported." % (depth, data_set))
            return

        self.inference  = tf.nn.softmax(logits, name = "softmax")
        self.predict    = tf.argmax(self.inference, 1)

        labels          = tf.reshape(tf.cast(self.labels, tf.int64), [-1])
        self.loss       = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        self.total_loss = tf.losses.get_total_loss()

        global_step     = tf.Variable(0, trainable=False)
        init_lr         = 0.1
        learning_rate   = tf.train.exponential_decay(init_lr, global_step, 1000, 0.96, staircase = True)

        update_ops      = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops) :
            self.train  = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.total_loss, global_step = global_step)

        is_correct      = tf.equal(self.predict, labels)
        self.accuracy   = tf.reduce_mean(tf.cast(is_correct, "float"), name = "accuracy")

        self.init = tf.global_variables_initializer()

        self.create_summary()

    def create_summary(self) :
        tf.summary.scalar(name = "loss",     tensor = self.loss)
        tf.summary.scalar(name = "accuracy", tensor = self.accuracy)

    def residual_stack(self, x, internel_channels, out_channels, first_stride, n_blocks, use_bottleneck, is_training, name_prefix) :
        for i in range(97, 97 + n_blocks) :
            if i > 122 :
                i -= 58

            # branch1    
            short_cut = x
            strides   = (1, 1)

            if i == 97 :
                strides = (first_stride, first_stride)
                short_cut = tf.layers.conv2d(inputs                = short_cut,
                                             filters               = out_channels,
                                             kernel_size           = 1,
                                             strides               = strides,
                                             padding               = "same",
                                             data_format           = "channels_last",
                                             kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                             kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                             name                  = name_prefix + chr(i) + "_conv_branch1a")
                short_cut = tf.layers.batch_normalization(inputs   = short_cut,
                                                          axis     = -1,
                                                          training = is_training,
                                                          name     = name_prefix + chr(i) + "_bn_branch1a")

            # branch2
            if use_bottleneck :
                x = tf.layers.conv2d(inputs                = x,
                                     filters               = internel_channels,
                                     kernel_size           = 1,
                                     strides               = strides,
                                     padding               = "same",
                                     data_format           = "channels_last",
                                     kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                     kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                     name                  = name_prefix + chr(i) + "_conv_branch2a")
                x = tf.layers.batch_normalization(inputs   = x,
                                                  axis     = -1,
                                                  training = is_training,
                                                  name     = name_prefix + chr(i) + "_bn_branch2a")
                x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu_branch2a")

                x = tf.layers.conv2d(inputs                = x,
                                     filters               = internel_channels,
                                     kernel_size           = 3,
                                     strides               = (1, 1),
                                     padding               = "same",
                                     data_format           = "channels_last",
                                     kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                     kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                     name                  = name_prefix + chr(i) + "_conv_branch2b")
                x = tf.layers.batch_normalization(inputs   = x,
                                                  axis     = -1,
                                                  training = is_training,
                                                  name     = name_prefix + chr(i) + "_bn_branch2b")
                x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu_branch2b")

                x = tf.layers.conv2d(inputs                = x,
                                     filters               = out_channels,
                                     kernel_size           = 1,
                                     strides               = (1, 1),
                                     padding               = "same",
                                     data_format           = "channels_last",
                                     kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                     kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                     name                  = name_prefix + chr(i) + "_conv_branch2c")
                x = tf.layers.batch_normalization(inputs   = x,
                                                  axis     = -1,
                                                  training = is_training,
                                                  name     = name_prefix + chr(i) + "_bn_branch2c")
            else :
                x = tf.layers.conv2d(inputs                = x,
                                     filters               = internel_channels,
                                     kernel_size           = 3,
                                     strides               = strides,
                                     padding               = "same",
                                     data_format           = "channels_last",
                                     kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                     kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                     name                  = name_prefix + chr(i) + "_conv_branch2a")
                x = tf.layers.batch_normalization(inputs   = x,
                                                  axis     = -1,
                                                  training = is_training,
                                                  name     = name_prefix + chr(i) + "_bn_branch2a")
                x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu_branch2a")

                x = tf.layers.conv2d(inputs                = x,
                                     filters               = out_channels,
                                     kernel_size           = 3,
                                     strides               = (1, 1),
                                     padding               = "same",
                                     data_format           = "channels_last",
                                     kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                                     kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                                     name                  = name_prefix + chr(i) + "_conv_branch2b")
                x = tf.layers.batch_normalization(inputs   = x,
                                                  axis     = -1,
                                                  training = is_training,
                                                  name     = name_prefix + chr(i) + "_bn_branch2b")

            # merge branch1 and branch2
            x = tf.add(short_cut, x, name = name_prefix + chr(i) + "_elementwise_add")
            x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu")

        return x
    
    def build_resnet_imagenet(self, raw_input, is_training, n_blocks) :
        # Model graph: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
        # DataSet: http://image-net.org/challenges/LSVRC/2015/

        # cast input from uint8 to float32
        x = tf.cast(raw_input, tf.float32)

        # 1-st Conv Layer
        x = tf.layers.conv2d(inputs                = x,
                             filters               = 64,
                             kernel_size           = 7,
                             strides               = (2, 2),
                             padding               = "same",
                             data_format           = "channels_last",
                             kernel_initializer    = tf.truncated_normal_initializer(stddev = 0.1),
                             kernel_regularizer    = tf.contrib.layers.l2_regularizer(0.00005),
                             name                  = "conv1")
        x = tf.layers.batch_normalization(inputs   = x,
                                          axis     = -1,
                                          training = is_training,
                                          name     = "bn_conv1")
        x = tf.nn.relu(features = x, name = "relu_conv1")

        x = tf.layers.max_pooling2d(inputs         = x,
                                    pool_size      = 3,
                                    strides        = 2,
                                    padding        = "same",
                                    data_format    = "channels_last",
                                    name           = "pool1")

        # conv2x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 64,
                                out_channels       = 256,
                                first_stride       = 1,
                                n_blocks           = n_blocks[0],
                                use_bottleneck     = True,
                                is_training        = is_training,
                                name_prefix        = "res2")
        
        # conv3_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 128,
                                out_channels       = 512,
                                first_stride       = 2,
                                n_blocks           = n_blocks[1],
                                use_bottleneck     = True,
                                is_training        = is_training,
                                name_prefix        = "res3")
        # conv4_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 256,
                                out_channels       = 1024,
                                first_stride       = 2,
                                n_blocks           = n_blocks[2],
                                use_bottleneck     = True,
                                is_training        = is_training,
                                name_prefix        = "res4")
        # conv5_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 512,
                                out_channels       = 2048,
                                first_stride       = 2,
                                n_blocks           = n_blocks[3],
                                use_bottleneck     = True,
                                is_training        = is_training,
                                name_prefix        = "res5")

        # global average pooling
        x = tf.layers.average_pooling2d(inputs      = x,
                                        pool_size   = 7,
                                        strides     = 1,
                                        padding     = "valid",
                                        data_format = "channels_last",
                                        name        = "pool5")

        # FC Layer
        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs = x,
                            units  = 1000,
                            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                            name   = "fc1000")
        return x

    def build_resnet_cifar(self, raw_input, is_training, n_blocks) :
        # DataSet: Cifar-10

        # cast input from uint8 to float32
        x = tf.cast(raw_input, tf.float32)

        # 1-st Conv Layer
        x = tf.layers.conv2d(inputs      = x,
                             filters     = 16,
                             kernel_size = 3,
                             strides     = (1, 1),
                             padding     = "same",
                             data_format = "channels_last",
                             kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                             name        = "conv1")
        x = tf.layers.batch_normalization(inputs   = x,
                                          axis     = -1,
                                          training = is_training,
                                          name     = "bn_conv1")
        x = tf.nn.relu(features = x, name = "relu_conv1")

        # conv2x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 16,
                                out_channels       = 16,
                                first_stride       = 1,
                                n_blocks           = n_blocks,
                                use_bottleneck     = False,
                                is_training        = is_training,
                                name_prefix        = "res2")

        # conv3_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 32,
                                out_channels       = 32,
                                first_stride       = 2,
                                n_blocks           = n_blocks,
                                use_bottleneck     = False,
                                is_training        = is_training,
                                name_prefix        = "res3")

        # conv4_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 64,
                                out_channels       = 64,
                                first_stride       = 2,
                                n_blocks           = n_blocks,
                                use_bottleneck     = False,
                                is_training        = is_training,
                                name_prefix        = "res4")

        # Global average pooling
        x = tf.layers.average_pooling2d(inputs      = x,
                                        pool_size   = 8,
                                        strides     = 1,
                                        padding     = "valid",
                                        data_format = "channels_last",
                                        name        = "pool5")

        # FC Layer
        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs = x,
                            units  = 10,
                            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                            name   = "fc10")

        return x

if __name__ == "__main__" :
    with tf.variable_scope("Resnet-18") :
        resnet18 = ResNet("ImageNet", 18)
    with tf.variable_scope("Resnet-50") :
        resnet50 = ResNet("ImageNet", 50)
    with tf.variable_scope("Resnet-152") :
        resnet152 = ResNet("ImageNet", 152)

    with tf.variable_scope("Resnet-20") :
        resnet20 = ResNet("CIFAR-10", 20)
    with tf.variable_scope("Resnet-56") :
        resnet56 = ResNet("CIFAR-10", 56)
    with tf.variable_scope("Resnet-110") :
        resnet110 = ResNet("CIFAR-10", 110)

