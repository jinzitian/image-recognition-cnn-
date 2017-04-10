# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:25:31 2017

@author: Jinzitian
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import  numpy as np
import tensorflow as tf

from model import model

tf.app.flags.DEFINE_integer('batch_size', 16, 'batch_size')
tf.app.flags.DEFINE_float('init_scale', 0.1, 'init_scale')
tf.app.flags.DEFINE_integer('number_class', 6, 'number_class')
tf.app.flags.DEFINE_integer('nb_epoch', 30, 'nb_epoch')
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'cnn_model', 'model save filename.')

FLAGS = tf.app.flags.FLAGS

def get_data(train_file):
    with open(train_file) as f:
        image_file_list = f.readlines()
    image_file_list = [i.strip() for i in image_file_list]
    processed_image = [] 
    labels = []
    for i in image_file_list:
        try:
            with open('./label'+'/'+i[:-3]+'txt') as f:
                label = f.readlines()
            x1,y1,x2,y2 = [int(j) for j in label[2].strip().split(' ')]
            image_all = Image.open('./image'+'/'+i)
            image_array = np.array(image_all)[x1:x2,y1:y2]
            image = Image.fromarray(image_array)
            re_image = np.array(image.resize((128,128)))
            processed_image.append(re_image)
            labels.append(int(label[0].strip()) if int(label[0].strip()) != -1 else 0)
            if int(label[0].strip()) == -1:
                print('there is a -1')
        except Exception as e:
            print('I found it %s'%i)
    images = np.concatenate(tuple(processed_image)).reshape((-1,128,128,3)).astype(np.float32)
    labels = np.array(labels)    
    return images,labels


def train():
    
    x_test_images, y_test_labels = get_data('./train_test/test.txt')
    x_test_images = x_test_images[:500]
    y_test_labels = y_test_labels[:500]
    x_images, y_labels = get_data('./train_test/train.txt')
    s = np.random.permutation(len(y_labels))
    x_images = x_images[s]
    y_labels = y_labels[s]
	
    train_num = x_images.shape[0]
    batch_num = train_num // FLAGS.batch_size
    
    loss_acc = []
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

        images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 128, 128, 3])
        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])                
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = model(images, labels, FLAGS.number_class)     
			
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            m_test = model(x_test_images, None, FLAGS.number_class)
            y_pre_tf = tf.argmax(m_test.predict,1)
			
        saver = tf.train.Saver()         
        with tf.Session() as session:
            session.run(tf.global_variables_initializer()) 
            for j in range(FLAGS.nb_epoch): 
                for i in range(batch_num):
                    _, loss = session.run([m.train_op, m.cost], feed_dict = {images: x_images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], labels: y_labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]})                              
                    if i%50 == 0:
                        y_pre = session.run(y_pre_tf)
                        accuracy = np.mean(y_pre == y_test_labels)
                        loss_acc.append((loss, accuracy))
                        print("Epoch: %d batch_num: %d loss: %.3f, accuracy = %s" % (j, i, loss, accuracy))
            #保存模型时一定注意保存的路径必须是英文的，中文会报错
            save_path = saver.save(session, FLAGS.checkpoints_dir + '/'+ FLAGS.model_prefix)
            print("Model saved in file: ", save_path)
            
    loss, acc = list(zip(*loss_acc))
    
    f = plt.figure()
    axes_loss = f.add_subplot(2,1,1)
    axes_acc = f.add_subplot(2,1,2)
    axes_loss.plot(loss)
    axes_loss.set_title('loss')
    axes_acc.plot(acc)
    axes_acc.set_title('accuracy')
    f.tight_layout(pad = 3)
    f.savefig('loss_acc.jpg', dpi = 500)
    f.show()            
            

    