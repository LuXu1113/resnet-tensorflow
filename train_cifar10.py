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
    graph = resnet.ResNet("CIFAR-10", 20)

    # 初始化网络参数
    LOG("Initializing graph ...")
    sess.run(graph.init)

    # 创建 saver 保存训练出的模型
    saver = tf.train.Saver(tf.global_variables())

    # 创建 summary_writer，保存训练过程中记录下来的 summary，在 tensorboard 中查看
    summary_writer = tf.summary.FileWriter(os.path.join(os.path.abspath("./summary")), sess.graph)
    summary_op = tf.summary.merge_all()

    LOG("Training ...")
    itr = 0
    for epoch in range(1, 250) :
        n_examples = len(x_train)
        batch_size = 128
        n_batch    = int((n_examples + batch_size - 1) / batch_size)
        loss       = 0.0

        # shuffle batch
        batch_list = [i for i in range(n_batch)]
        random.shuffle(batch_list)

        # Train all batches.
        train_acc  = 0.0
        train_loss = 0.0
        LOG("epoch-%d:" % epoch)
        start_time = time.time()
        for batch_no in batch_list :
            x_ = x_train[batch_no * batch_size : (batch_no + 1) * batch_size]
            y_ = y_train[batch_no * batch_size : (batch_no + 1) * batch_size]
            summary, loss, acc, _  = sess.run([summary_op, graph.total_loss, graph.accuracy, graph.train], feed_dict = {graph.instances : x_, graph.labels: y_, graph.is_training: True})

            summary_writer.add_summary(summary, sess.run(graph.step))
            train_acc  += len(x_) * acc
            train_loss += len(x_) * loss
        finish_time = time.time()

        # Print performance and accurcy
        duration = finish_time - start_time
        examples_per_sec = n_examples / duration
        sec_per_batch = duration / n_batch
        train_acc  /= n_examples / 100.0
        train_loss /= n_examples
        LOG("[ perf     ]: %.1f examples/sec, %.3f sec/batch" % (examples_per_sec, sec_per_batch))
        LOG("[ train    ]: accuracy = %.2f%%, loss = %.2lf" % (train_acc, train_loss))

        # Compute accuracy on validate set
        n_batch = int((len(x_validate) + batch_size - 1) / batch_size)
        validate_acc = 0.0
        for batch_no in range(n_batch) :
            x_ = x_validate[batch_no * batch_size : (batch_no + 1) * batch_size]
            y_ = y_validate[batch_no * batch_size : (batch_no + 1) * batch_size]
            acc = sess.run(graph.accuracy, feed_dict = {graph.instances: x_, graph.labels: y_, graph.is_training: False})
            validate_acc += acc * len(x_)

        validate_acc /= len(x_validate) / 100.0
        LOG("[ validate ]: accuracy = %.2f%%" % validate_acc)

        if epoch % 13 == 0 :
            saver.save(sess, os.path.join(os.path.abspath("./model"), "model"), global_step = sess.run(graph.step))
            
    sess.close()
    log_fh.close()
