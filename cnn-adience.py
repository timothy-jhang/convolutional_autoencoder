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

from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D, Conv2Dtranspose
from my_nn_lib import FullConnected, ReadOutLayer

# Utilize Keras ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

# Adience Age Classification Model 
def mk_nn_model(x, y_):
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

    drop6 = tf.nn.dropout(fc6_out,keep_prob=0.5)

    fc7 = FullConnected(drop6, 512, 512, activation='relu')
    fc7_out = fc7.output()

    drop7 = tf.nn.dropout(fc7_out,keep_prob=0.5)

    fc8 = FullConnected(drop7, 512, 8, activation='relu')
    fc8_out = fc8.output()

    # crossentry for  classifier
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc8_out)
    loss = tf.reduce_mean(cross_entropy)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(fc8_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, fc8_out


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    # Variables
    x = tf.placeholder(tf.float32, [None, 227,227,3])
    y_ = tf.placeholder(tf.float32, [None, 8]) 

    loss, accuracy, fo = mk_nn_model(x, y_)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    fold = './fold0'
    training_set = train_datagen.flow_from_directory(fold + '/age_train', target_size = (227, 227), batch_size = 128, class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(fold + '/age_test', target_size = (227, 227), batch_size = 128, class_mode = 'categorical')

    # initializer
    init = tf.global_variables_initializer()
    
    # tuning
    print("Fine Tuning weights for classification tasks...")
    with tf.Session() as sess:
    	# Initialize variables
    	sess.run(init)
        for i in range(30001):
            batch_xs, batch_ys = training_set.next()
            train_step.run({x: batch_xs, y_: batch_ys})
            if i % 1000 == 0:
                train_loss= loss.eval({x: batch_xs, y_: batch_ys})
                train_acc= accuracy.eval({x: batch_xs, y_: batch_ys})
                print('  Train: step, loss, accuracy = ',i, train_loss, train_acc)
		tbatch_xs, tbatch_ys = test_set.next()
		test_fd = {x: tbatch_xs, y_: tbatch_ys }
		print('  Test: loss, accuracy = ', loss.eval(test_fd), accuracy.eval(test_fd))
		print('fo_out = ', fo[0], fo[0].shape)

	# Save classifier model weights to disk
	saver1 = tf.train.Saver()
	save_path1 = saver.save(sess, './model/classifier-model.ckt')
	print("Classifier Model saved in file: %s" % save_path)
 
