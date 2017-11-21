# encoding: utf-8
# python-version: 3.6

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

class ResNet :
    def __init__(self, data_set, depth) :
        # with tf.variable_scope("Resnet%d_%s" % (depth, data_set)) :
        self.is_training = tf.placeholder(tf.bool, [], name = "is_training")

        self.instances = None
        self.labels    = None
        self.inference = None
        logits = None

        if data_set == "CIFAR-10" :
            self.instances = tf.placeholder(tf.uint8, [None, 32, 32, 3], name = "instances")
            self.labels    = tf.placeholder(tf.uint8, [None, 1], name = "labels")

            if depth == 20 :
                logits = self.build_resnet_20(self.instances, self.is_training)

        elif data_set == "ImageNet" :
            self.instances = tf.placeholder(tf.uint8, [None, 224, 224, 3], name = "instances")
            self.labels    = tf.placeholder(tf.uint8, [None, 1], name = "labels")

            if depth == 50 :
                logits = self.build_resnet_50(self.instances, self.is_training)

        if None == logits:
            print("Resnet%d for dataset %s is not supported." % (depth, data_set))
            return

        self.inference  = tf.nn.softmax(logits, name = "softmax")
        self.predict    = tf.argmax(self.inference, 1)

        labels          = tf.cast(self.labels, tf.int64)
        self.loss       = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        self.total_loss = tf.losses.get_total_loss()
        self.train      = tf.train.MomentumOptimizer(0.1, 0.9).minimize(self.total_loss)

        is_correct      = tf.equal(self.predict, labels)
        self.accuracy   = tf.reduce_mean(tf.cast(is_correct, "float"))

        self.init = tf.global_variables_initializer()

    def residual_stack(self, x, internel_channels, out_channels, first_stride, n_stacks, is_training, name_prefix) :
        for i in range(97, 97 + n_stacks) :
            # branch1    
            short_cut = x
            strides   = (1, 1)

            if i == 97 :
                strides = (first_stride, first_stride)
                short_cut = tf.layers.conv2d(inputs      = short_cut,
                                             filters     = out_channels,
                                             kernel_size = 1,
                                             strides     = strides,
                                             padding     = "same",
                                             data_format = "channels_last",
                                             kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                                             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                                             name        = name_prefix + chr(i) + "_conv_branch1a")
                short_cut = tf.layers.batch_normalization(inputs   = short_cut,
                                                          axis     = -1,
                                                          training = is_training,
                                                          name     = name_prefix + chr(i) + "_bn_branch1a")

            # branch2
            x = tf.layers.conv2d(inputs      = x,
                                 filters     = internel_channels,
                                 kernel_size = 1,
                                 strides     = strides,
                                 padding     = "same",
                                 data_format = "channels_last",
                                 kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                                 name        = name_prefix + chr(i) + "_conv_branch2a")
            x = tf.layers.batch_normalization(inputs   = x,
                                              axis     = -1,
                                              training = is_training,
                                              name     = name_prefix + chr(i) + "_bn_branch2a")
            x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu_branch2a")

            x = tf.layers.conv2d(inputs      = x,
                                 filters     = internel_channels,
                                 kernel_size = 3,
                                 strides     = (1, 1),
                                 padding     = "same",
                                 data_format = "channels_last",
                                 kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                                 name        = name_prefix + chr(i) + "conv_branch2b")
            x = tf.layers.batch_normalization(inputs   = x,
                                              axis     = -1,
                                              training = is_training,
                                              name     = name_prefix + chr(i) + "_bn_branch2b")
            x = tf.nn.relu(features = x, name = name_prefix + chr(i) + "_relu_branch2b")

            x = tf.layers.conv2d(inputs      = x,
                                 filters     = out_channels,
                                 kernel_size = 1,
                                 strides     = (1, 1),
                                 padding     = "same",
                                 data_format = "channels_last",
                                 kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                                 name        = name_prefix + chr(i) + "conv_branch2c")
            x = tf.layers.batch_normalization(inputs   = x,
                                              axis     = -1,
                                              training = is_training,
                                              name     = name_prefix + chr(i) + "_bn_branch2c")

            # merge branch1 and branch2
            x = tf.nn.relu(features = short_cut + x, name = name_prefix + chr(i) + "relu")

        return x
    
    def build_resnet_50(self, raw_input, is_training) :
        # Model graph: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
        # DataSet: http://image-net.org/challenges/LSVRC/2015/

        # cast input from uint8 to float32
        x = tf.cast(raw_input, tf.float32)

        # 1-st Conv Layer
        x = tf.layers.conv2d(inputs      = x,
                             filters     = 64,
                             kernel_size = 7,
                             strides     = (2, 2),
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

        x = tf.layers.max_pooling2d(inputs      = x,
                                    pool_size   = 3,
                                    strides     = 2,
                                    padding     = "same",
                                    data_format = "channels_last",
                                    name        = "pool1")

        # conv2x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 64,
                                out_channels       = 256,
                                first_stride       = 1,
                                n_stacks           = 3,
                                is_training        = is_training,
                                name_prefix        = "res2")
        # conv3_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 128,
                                out_channels       = 512,
                                first_stride       = 2,
                                n_stacks           = 4,
                                is_training        = is_training,
                                name_prefix        = "res3")
        # conv4_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 256,
                                out_channels       = 1024,
                                first_stride       = 2,
                                n_stacks           = 6,
                                is_training        = is_training,
                                name_prefix        = "res4")
        # conv5_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 512,
                                out_channels       = 2048,
                                first_stride       = 2,
                                n_stacks           = 3,
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
                            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                            name   = "fc1000")
        return x

    def build_resnet_20(self, raw_input, is_training) :
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
                                n_stacks           = 2,
                                is_training        = is_training,
                                name_prefix        = "res2")

        # conv3_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 32,
                                out_channels       = 32,
                                first_stride       = 2,
                                n_stacks           = 2,
                                is_training        = is_training,
                                name_prefix        = "res3")

        # conv4_x
        x = self.residual_stack(x                  = x,
                                internel_channels  = 64,
                                out_channels       = 64,
                                first_stride       = 2,
                                n_stacks           = 2,
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
                            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00005),
                            name   = "fc10")

        return x
                            
                            

if __name__ == "__main__" :
    with tf.variable_scope("Resnet-50") :
        resnet = ResNet("ImageNet", 50)
    with tf.variable_scope("Resnet-20") :
        resnet = ResNet("CIFAR-10", 20)

