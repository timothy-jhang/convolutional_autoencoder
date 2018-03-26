#
# autoencoder convolutional   (tensorflow version)
# 2018.03.21.
# caffe version -> tensorflow  (v1)
# Caffe version (paper): A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe (Volodymyr Turchenko, Eric Chalmers, Artur Luczak)
# caffe version has no detailed connection from full connected layer to the next deconv. layer, so I added image resizing
# caffe version (website) : https://groups.google.com/forum/#!topic/caffe-users/GhrCtONcRxY
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from my_nn_lib import Convolution2D, MaxPooling2D
from my_nn_lib import FullConnected, ReadOutLayer


# Up-sampling 2-D Layer (deconvolutoinal Layer)
class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''
    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='sigmoid',stride=1):
        self.input = input      
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.out_ch = out_ch
        self.activation = activation
        
        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]    # note the arguments order
	        
        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
                            trainable=True)
        b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
                            trainable=True)
        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]
        self.s = stride 
    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]      
        linout = tf.nn.conv2d_transpose(value=self.input, filter=self.w, output_shape=shape4D,
			strides=[1, self.s, self.s, 1]) + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout
	print('w=',self.w.shape,'output_shape=',self.batsiz,self.rows,self.cols,self.out_ch,'Conv2DTr-shape-',self.output.shape)        
        return self.output

# Create the model
def model(X, w_e, b_e, w_d, b_d):
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)
    
    return encoded, decoded

def mk_nn_model(x, y_):
    # Encoding phase
    x_image = tf.reshape(x, [-1, 28, 28, 1])    
    conv1 = Convolution2D(x_image, (28, 28), 1, 8, 
                          (9, 9), activation='sigmoid')
    conv1_out = conv1.output()

#    pool1 = MaxPooling2D(conv1_out)
#    pool1_out = pool1.output()
#    pool1_out = tf.nn.dropout(pool1_out,keep_prob=0.2)

    conv2 = Convolution2D(conv1_out, (28, 28), 8, 4, 
                          (9, 9), activation='sigmoid')
    conv2_out = conv2.output()
    
#    pool2 = MaxPooling2D(conv2_out)
#    pool2_out = pool2.output()
#    pool2_out = tf.nn.dropout(pool2_out,keep_prob=0.2)

    # at this point the representation is (4, 28, 28) i.e. 128*16-dimensional
    po = tf.reshape(conv2_out,[-1,4*28*28])

    fc = FullConnected(po, 4*28*28, 256, activation='sigmoid')
    fc_out = fc.output()

#    fc2 = FullConnected(fc_out, 256, 2, activation='sigmoid')
#    fc2_out = fc2.output()

    fo = FullConnected(fc_out, 256, 10, activation='sigmoid')
    fo_out = fo.output()

    # Decoding phase
    dfc1 = FullConnected(fo_out, 10, 256, activation='sigmoid')
    dfc1_out = dfc1.output()

# reshape 
    deconvin = tf.reshape(dfc1_out, [-1,16,16,1])

#resize_images(images, size, method=ResizeMethod.BILINEAR, align_corners=False)
    deconvin = tf.image.resize_images(deconvin, (28,28),method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    conv_t1 = Conv2Dtranspose(deconvin, (28, 28), 1, 4,
                         (12, 12), activation='sigmoid')
    conv_t1_out = conv_t1.output()

    conv_t2 = Conv2Dtranspose(conv_t1_out, (28, 28), 4, 4,
                         (17, 17), activation='sigmoid')
    conv_t2_out = conv_t2.output()

    conv_t3 = Conv2Dtranspose(conv_t2_out, (28, 28), 4, 1, 
                         (1, 1), activation='sigmoid')
    decoded = conv_t3.output()

    decoded = tf.reshape(decoded, [-1, 784])
    cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)

    # crossentry for  classifier
    cross_entropy_acc = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fo_out)
    lossacc = tf.reduce_mean(cross_entropy_acc)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(fo_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, decoded, lossacc, fo_out, accuracy


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) 

    loss, decoded, lossacc, ro, accuracy = mk_nn_model(x, y_)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    acc_step = tf.train.AdagradOptimizer(0.1).minimize(lossacc)
    init = tf.global_variables_initializer()

    print(' Direct tuning of weights ... ')
    with tf.Session() as sess:
    	# Initialize variables
    	sess.run(init)

# Save classifier model weights to disk
	saver1 = tf.train.Saver(max_to_keep=5)
	best_acc = 0.0
#    	print("Trained Model restored from file: %s" % save_path)
        for i in range(50001):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            acc_step.run({x: batch_xs, y_: batch_ys})
            if i % 1000 == 0:
                train_loss= lossacc.eval({x: batch_xs, y_: batch_ys})
                train_acc= accuracy.eval({x: batch_xs, y_: batch_ys})
                print('  step, loss, accuracy = ',i, train_loss, train_acc)
	    if best_acc < train_acc :
		best_acc = train_acc
		save1_path = saver1.save(sess, './model/best5-classifier-model.ckt')
		print("Classifier Model saved in file: %s" % save1_path, 'acc=', train_acc)


        # generate decoded image with test data
        test_fd = {x: mnist.test.images, y_: mnist.test.labels}
        ro_imgs = ro.eval(test_fd)
        print('cross entry loss (test) = ', lossacc.eval(test_fd))
        print('accuracy (test) = ', accuracy.eval(test_fd))
        print('ro_imgs = ', ro_imgs[0], ro_imgs[0].shape)

 
    x_test = mnist.test.images
    n = 10  # how many digits we will display
    for i in range(n):
	print('i=',i,np.argmax(ro_imgs[i]))
    
