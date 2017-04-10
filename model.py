# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:25:31 2017

@author: Jinzitian
"""

import tensorflow as tf
import numpy as np

class model(object):

    def __init__(self, images, label, number_class):
        
        if isinstance(images, tf.Tensor):
            batch_size = images.shape[0].value
            print('train, batch_size is %s'%batch_size)
        elif isinstance(images, np.ndarray):
            batch_size = images.shape[0]
            print('predict, batch_size is %s'%batch_size)
        else:
            print('error')
            return
        # conv1
        kernel1 = tf.get_variable('weights1',shape=[5, 5, 3, 64], dtype=tf.float32)
        conv_1 = tf.nn.conv2d(images, kernel1, [1, 1, 1, 1], padding='VALID')
        biases1 = tf.get_variable('biases1', [64], dtype=tf.float32)
        pre_activation1 = tf.nn.bias_add(conv_1, biases1)
        conv1 = tf.nn.relu(pre_activation1)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='VALID', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        # conv2
        kernel2 = tf.get_variable('weights2',shape=[5, 5, 64, 64], dtype=tf.float32)
        conv_2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='VALID')
        biases2 = tf.get_variable('biases2', [64], dtype=tf.float32)
        pre_activation2 = tf.nn.bias_add(conv_2, biases2)
        conv2 = tf.nn.relu(pre_activation2)
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        # local3
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights3 = tf.get_variable('weights3', shape=[dim, 384], dtype=tf.float32)
        biases3 = tf.get_variable('biases3', [384], dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights3) + biases3)
        # local4
        weights4 = tf.get_variable('weights4', shape=[384, 192], dtype=tf.float32)
        biases4 = tf.get_variable('biases4', [192], dtype=tf.float32)
        local4 = tf.nn.relu(tf.matmul(local3, weights4) + biases4)
        # last
        weights5 = tf.get_variable('weights5', [192, number_class], dtype=tf.float32)
        biases5 = tf.get_variable('biases5', [number_class], dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(local4, weights5), biases5)
        predict = tf.nn.softmax(softmax_linear)
        self.predict = predict
        if label is not None:
            targets = tf.one_hot(tf.reshape(label, [-1]), depth=number_class)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits=softmax_linear)
            cost = tf.reduce_mean(loss)
            self.cost = cost
            train_op = tf.train.AdamOptimizer(0.01).minimize(cost)
            self.train_op = train_op

