# encoding: utf-8
# python-version: 3.6

from datetime import datetime
import time
import tensorflow as tf
import resnet
import os

def LOG(str) :
    print("%s: %s" % (datetime.now(), str))

if __name__ == "__main__" :
    (x_train, y_train), (x_validate, y_validate) = tf.keras.datasets.cifar10.load_data()
    
    sess = tf.Session()

    # 构建计算流图
    LOG("Building graph ...")
    graph = resnet.ResNet("CIFAR-10", 20)

    # 初始化网络参数
    LOG("Initializing graph ...")
    sess.run(graph.init)

    # 创建 saver 保存训练出的模型
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(os.path.join(os.path.abspath("./summary")), sess.graph)
    summary_writer.add_graph(sess.graph)

    LOG("Training ...")
    for epoch in range(1, 200) :
        n_examples = len(x_train)
        batch_size = 32
        n_batch    = int((n_examples + batch_size - 1) / batch_size)
        loss       = 0.0

        # Train all batches.
        start_time = time.time()
        for batch_no in range(n_batch) :
            x_ = x_train[batch_no * batch_size : (batch_no + 1) * batch_size]
            y_ = y_train[batch_no * batch_size : (batch_no + 1) * batch_size]
            summary, _  = sess.run([tf.summary.merge_all(), graph.train], feed_dict = {graph.instances : x_, graph.labels: y_, graph.is_training: True})

            if (batch_no + 1) % 30 == 0 :
                LOG("%d/%d\tbatches trained." % (batch_no + 1, n_batch))
                summary_writer.add_summary(summary, (epoch - 1) * n_batch + batch_no)

        finish_time = time.time()
        LOG("%d/%d\tbatches trained." % (n_batch, n_batch))

        # Compute performance
        duration = finish_time - start_time
        examples_per_sec = n_examples / duration
        sec_per_batch = duration / n_batch

        LOG("Testing ...")
        # Compute accuracy on validate set
        n_batch = int((len(x_validate) + batch_size - 1) / batch_size)
        accuracy = 0.0
        for batch_no in range(n_batch) :
            x_ = x_validate[batch_no * batch_size : (batch_no + 1) * batch_size]
            y_ = y_validate[batch_no * batch_size : (batch_no + 1) * batch_size]
            acc = sess.run(graph.accuracy, feed_dict = {graph.instances: x_, graph.labels: y_, graph.is_training: False})
            accuracy += acc * len(x_)
            
        accuracy /= len(x_validate)
        LOG("epoch-%d: accuracy = %.4f%% (%.1f examples/sec; %.3f sec/batch)" % (epoch, accuracy * 100, examples_per_sec, sec_per_batch))

        if epoch % 13 == 0 :
            saver.save(sess, os.path.join(os.path.abspath("./model"), "resnet20_epoch%d_" % epoch))
            

