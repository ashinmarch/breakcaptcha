## Sync with Houdong about debugging
# 1. Initial cost function value is 5000, not reasonable. Debug the initial cost function.
#    Reasonable value should be like 1*log(1/10) * 4 ~= 15. Resolved by multiply input matrix by 1/255.0
# 2. Try 1 digit first

import argparse
import random
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from io import BytesIO
from captcha.image import ImageCaptcha
from subprocess import call
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def deepnn_orig(x, h, w):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, h, w, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Third convolutional layer
  W_conv3 = weight_variable([3, 3, 64, 64])
  b_conv3 = bias_variable([64])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Third pooling layer.
  h_pool3 = maxpool2d(h_conv3)

  h_new = (int)((h + 3) / 4)
  w_new = (int)((w + 3) / 4)
  W_fc1 = weight_variable([h_new * w_new * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool3_flat = tf.reshape(h_pool3, [-1, h_new * w_new * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc21 = weight_variable([1024, 10])
  b_fc21 = bias_variable([10])
  y_conv21 = tf.matmul(h_fc1_drop, W_fc21) + b_fc21

  W_fc22 = weight_variable([1024, 10])
  b_fc22 = bias_variable([10])
  y_conv22 = tf.matmul(h_fc1_drop, W_fc22) + b_fc22

  W_fc23 = weight_variable([1024, 10])
  b_fc23 = bias_variable([10])
  y_conv23 = tf.matmul(h_fc1_drop, W_fc23) + b_fc23

  W_fc24 = weight_variable([1024, 10])
  b_fc24 = bias_variable([10])
  y_conv24 = tf.matmul(h_fc1_drop, W_fc24) + b_fc24

  return y_conv21, y_conv22, y_conv23, y_conv24, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def maxpool2d(x, k = 2, s = 1):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

def avgpool2d(x, k = 2, s = 1):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def digit_seq_gen(digit):
    return [1 if i == digit else 0 for i in range(10)]

def get_label(num):
    arr = []
    for _ in range(4):
        digit_seq = digit_seq_gen(num % 10)
        arr.insert(0, digit_seq)
        num = int(num / 10)

    # 4 * 10
    return arr


def gen_rand():
    return random.randint(0, 9999)
    
def get_image_gen(captcha, num, h, w):
    numstr = '{0:04}'.format(num)
    img = captcha.generate(numstr)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (w, h))

    img = np.multiply(img, 1/255.0)
    img = np.reshape(img, -1)

    return img

def next_batch(captcha, batch_size, h, w):
    data = []
    label = []
    for i in range(batch_size):
        num = gen_rand() 
        img = get_image_gen(captcha, num, h, w)

        # -1 * pixels, pixels = 60 * 160
        data.append(img)

        # -1 * 4 * 10
        label.append(get_label(num))

    return data, np.asarray(label)

def get_data(captcha, num, h, w):
    img = get_image_gen(captcha, num, h, w)
    return np.array([img])

def calc_accuracy(pred1, label1, pred2, label2, pred3, label3, pred4, label4):
    pred1_argmax = tf.argmax(pred1, 1)
    label1_argmax = tf.argmax(label1, 1)
    pred2_argmax = tf.argmax(pred2, 1)
    label2_argmax = tf.argmax(label2, 1)
    pred3_argmax = tf.argmax(pred3, 1)
    label3_argmax = tf.argmax(label3, 1)
    pred4_argmax = tf.argmax(pred4, 1)
    label4_argmax = tf.argmax(label4, 1)
    equal_arr1 = tf.equal(pred1_argmax, label1_argmax)
    equal_arr2 = tf.equal(pred2_argmax, label2_argmax)
    equal_arr3 = tf.equal(pred3_argmax, label3_argmax)
    equal_arr4 = tf.equal(pred4_argmax, label4_argmax)

    correct_pred = tf.logical_and(tf.logical_and(equal_arr1, equal_arr2), tf.logical_and(equal_arr3, equal_arr4))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy

w = 80
h = 30
lr = 1e-3

# Create the model
x = tf.placeholder(tf.float32, [None, h * w])

# Define loss and optimizer
y1_ = tf.placeholder(tf.float32, [None, 10])
y2_ = tf.placeholder(tf.float32, [None, 10])
y3_ = tf.placeholder(tf.float32, [None, 10])
y4_ = tf.placeholder(tf.float32, [None, 10])

# Build the graph for the deep net
y_conv1, y_conv2, y_conv3, y_conv4, keep_prob = deepnn_orig(x, h, w)
#y_conv1, y_conv2, y_conv3, y_conv4, keep_prob = deepnn(x, h, w)

# Split to 4 steps for 4 digits
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1_, logits=y_conv1))
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2_, logits=y_conv2))
cross_entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y3_, logits=y_conv3))
cross_entropy4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y4_, logits=y_conv4))

cross_entropy_sum = tf.reduce_sum([cross_entropy1, cross_entropy2, cross_entropy3, cross_entropy4], reduction_indices=[0])

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy_sum)
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

accuracy = calc_accuracy(y_conv1, y1_, y_conv2, y2_, y_conv3, y3_, y_conv4, y4_)

saver = tf.train.Saver()

with tf.Session() as sess:
    captcha = ImageCaptcha(fonts=['arial.ttf'])

    # data for evaluation
    data_test, label_test = next_batch(captcha, 1000, h, w)
    label1_test = label_test[:, 0, :]
    label2_test = label_test[:, 1, :]
    label3_test = label_test[:, 2, :]
    label4_test = label_test[:, 3, :]

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--task', choices = ['train', 'eval'], help='Task type: train or eval', required=True)
    parser.add_argument('integer', type=int, nargs='?', help='an integer for evaluate')
    args = parser.parse_args()

    if args.task.lower() == 'train':
        # Train
        sess.run(tf.global_variables_initializer())

        for epoch in range(4):
            for i in range(2000):
              data, label = next_batch(captcha, 50, h, w)

              # labels
              label1 = label[:, 0, :]
              label2 = label[:, 1, :]
              label3 = label[:, 2, :]
              label4 = label[:, 3, :]

              #print(np.shape(data))
              #print(np.shape(label1))

              if i % 100 == 0:
                  dt = datetime.now().strftime("%Y-%m-%d %H:%M")
                  ce_val = cross_entropy_sum.eval(feed_dict={x: data, y1_: label1,
                      y2_: label2, y3_: label3, y4_: label4, keep_prob: 1.0})

                  # Use 1000 dataset to test accuracy
                  acc_val = accuracy.eval(feed_dict={x: data_test, y1_: label1_test,
                      y2_: label2_test, y3_: label3_test, y4_: label4_test, keep_prob: 1.0})

                  print("[%s] Epoch=[%d] step=[%d]  CrossEntropy=[%s] Accuracy=[%s]" % (dt, epoch, i, ce_val, acc_val))
              
              if i > 0 and i % 1000 == 0:
                  lr = lr / 2

              train_step.run(feed_dict={x: data, y1_: label1, y2_: label2, y3_: label3, y4_: label4, keep_prob: 0.7})

        # Save model
        save_path = saver.save(sess, "./models/break_pycaptcha_4d.ckpt")
        print("Model saved in file: %s" % save_path)

    else:
        # Evaluation accuracy and predict number
        saver.restore(sess, "./models/break_pycaptcha_4d.ckpt")

        print("Accuracy Testing: %s" % sess.run([accuracy], feed_dict={x: data_test,
            y1_: label1_test, y2_: label2_test, y3_: label3_test, y4_: label4_test, keep_prob: 1.0}))

        # Predict
        pred_int = args.integer
        if pred_int != None and pred_int in range(0, 10000):
            data_test = get_data(captcha, pred_int, h, w)
            print("%d%d%d%d" % (
                tf.argmax(y_conv1, 1).eval(feed_dict={x: data_test, keep_prob: 1.0}),
                tf.argmax(y_conv2, 1).eval(feed_dict={x: data_test, keep_prob: 1.0}),
                tf.argmax(y_conv3, 1).eval(feed_dict={x: data_test, keep_prob: 1.0}),
                tf.argmax(y_conv4, 1).eval(feed_dict={x: data_test, keep_prob: 1.0})
                ))
        else:
            print('Please enter a number in range [0, 9999]!')

