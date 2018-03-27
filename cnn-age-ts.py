#
# Caffe Adience CNN  in  tensorflow 
# 2018.03.23
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from my_nn_lib import Convolution2D, MaxPooling2D, Conv2Dtranspose
from my_nn_lib import FullConnected, ReadOutLayer

# Utilize Keras ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

# Adience Age Classification Model 
def mk_nn_model(x, y_, pkeep):
    # Encoding phase
    x_image = tf.reshape(x, [-1, 227, 227, 3])    

#1st conv. 
    conv1 = Convolution2D(x_image, (227, 227), 3, 96, (7, 7), activation='relu',S=4)
    conv1_out = conv1.output()

#1st pooling
    pool1 = MaxPooling2D(conv1_out,ksize=[1,3,3,1], S=2)
    pool1_out = pool1.output()

#dropout?
#    pool1_out = tf.nn.dropout(pool1_out,keep_prob=0.2)

#LRN1
    norm1 = tf.nn.local_response_normalization(pool1_out, depth_radius=5, alpha=0.0001, beta=0.75)
 
#2nd conv. 
    conv2 = Convolution2D(norm1, (29, 29), 96, 256, (5, 5), activation='relu',S=1)  # pad=2 - how to do it????
    conv2_out = conv2.output()

#2nd pooling    
    pool2 = MaxPooling2D(conv2_out, ksize=(1,3,3,1), S=2)
    pool2_out = pool2.output()
#    pool2_out = tf.nn.dropout(pool2_out,keep_prob=0.2)
    norm2 = tf.nn.local_response_normalization(pool2_out, depth_radius=5, alpha=0.0001, beta=0.75)

#3rd conv. 
    conv3 = Convolution2D(norm2, (15, 15), 256, 384, (3, 3), activation='relu',S=1)  # pad=2 - how to do it????
    conv3_out = conv3.output()

#3rd pooling
    pool3 = MaxPooling2D(conv3_out, ksize=(1,3,3,1), S=2)
    pool3_out = pool3.output()

    # at this point the representation is (4, 28, 28) i.e. 128*16-dimensional
    po = tf.reshape(pool3_out,[-1,384*8*8])

    fc6 = FullConnected(po, 384*8*8, 512, activation='relu')
    fc6_out = fc6.output()

    drop6 = tf.nn.dropout(fc6_out,keep_prob=pkeep)

    fc7 = FullConnected(drop6, 512, 512, activation='relu')
    fc7_out = fc7.output()

    drop7 = tf.nn.dropout(fc7_out,keep_prob=pkeep)

    fc8 = FullConnected(drop7, 512, 8, activation='relu')
    fc8_out = fc8.output()

    # crossentry for  classifier
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc8_out)
    loss = tf.reduce_mean(cross_entropy)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(fc8_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, fc8_out

def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = training_set.next()

    # learning rate decay
    max_learning_rate = 0.001
    min_learning_rate = 0.00001
    decay_speed = 10000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], {x: batch_X, y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
#        datavis.append_training_curves_data(i, a, c)
#        datavis.update_image1(im)
#        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    if update_test_data:
	tbatch_X, tbatch_Y = test_set.next()
        a, c = sess.run([accuracy, cross_entropy], {x: tbatch_X, y_: tbatch_Y, pkeep: 1.0})
        print(str(i) + ": ********* epoch " + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
#        datavis.append_test_curves_data(i, a, c)
#        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {x: batch_X, y_: batch_Y, lr: learning_rate, pkeep: 0.5})

if __name__ == '__main__':
    # Variables
    x = tf.placeholder(tf.float32, [None, 227,227,3])
    y_ = tf.placeholder(tf.float32, [None, 8]) 
    lr = tf.placeholder(tf.float32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)

    cross_entropy, accuracy, fo = mk_nn_model(x, y_, pkeep)

    # training step, the learning rate is a placeholder
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    fold = './fold0'
    training_set = train_datagen.flow_from_directory(fold + '/age_train', target_size = (227, 227), batch_size = 128, class_mode = 'categorical',shuffle=True)
    test_set = test_datagen.flow_from_directory(fold + '/age_test', target_size = (227, 227), batch_size = 128, class_mode = 'categorical',shuffle=True)

    # initializer
    init = tf.global_variables_initializer()

    # session
    sess = tf.Session()
    sess.run(init)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
    print('Training ......')
    for i in range(10000+1): training_step(i, i % 100 == 0, i % 50 == 0)

# Save classifier model weights to disk
#saver = tf.train.Saver()
#save_path = saver.save(sess, './model/cnnage-model.ckt')
#print("Classifier Model saved in file: %s" % save_path)
    
 
