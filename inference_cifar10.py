# encoding: utf-8
# python-version: 3.6

import os
import time
import math
import random
from datetime import datetime
import tensorflow as tf
import resnet

log_fh = None

def LOG(str, end = "\n") :
    print("%s: %s" % (datetime.now(), str), end = end)
    log_fh.write("%s: %s%s" % (datetime.now(), str, end))
    log_fh.flush()

if __name__ == "__main__" :
    log_fh = open("train.log", "w")
    
    if tf.__version__ > "1.3.0" :
        (x_train, y_train), (x_validate, y_validate) = tf.keras.datasets.cifar10.load_data()
    else :
        (x_train, y_train), (x_validate, y_validate) = tf.contrib.keras.datasets.cifar10.load_data()

    sess = tf.Session()

    # 构建计算流图
    LOG("Building graph ...")
    graph = resnet.ResNet(sess = sess, model = os.path.abspath("./model/model-96577"))

    # Compute accuracy on validate set
    x_ = [None]
    y_ = [None]
    err = 0
    for batch_no in range(len(x_validate)) :
        x_[0] = x_validate[batch_no]
        y_[0] = y_validate[batch_no]
        y  = sess.run(graph.predict, feed_dict = {graph.instances : x_, graph.is_training : False})

        if not (y_[0][0] == y[0]) :
            print(y_[0][0], y[0])
            err = err + 1
    print("err: ", err)
    sess.close()
    log_fh.close()
