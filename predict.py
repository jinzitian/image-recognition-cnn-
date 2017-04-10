# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:15:52 2017

@author: Jinzitian
"""

from model import *
from PIL import Image

from train import FLAGS 


def image_process(image_file_name):
    with open('./label'+image_file_name[7:-3]+'txt') as f:
        label = f.readlines()
    x1,y1,x2,y2 = [int(i) for i in label[2].strip().split(' ')]
    image_all = Image.open(image_file_name)
    image_array = np.array(image_all)[x1:x2,y1:y2]
    image = Image.fromarray(image_array)
    re_image = np.array(image.resize((128,128)))
    return np.array(re_image).reshape((1,128,128,3)).astype(np.float32)


def predict(image):
    
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
               
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = model(image, None, FLAGS.number_class)     
            y_pre_tf = tf.argmax(m.predict,1)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        saver = tf.train.Saver()         
        with tf.Session() as session:
            session.run(tf.global_variables_initializer()) 
            saver.restore(session, ckpt.model_checkpoint_path)
            y_pre = session.run(y_pre_tf)
            
        return y_pre[0] if y_pre[0]!=0 else -1
        
