# Train as a summation of multiple gaussian-poisson curvatures

import tensorflow as tf
import numpy as np
import nucdata
import sys,os

imh = nucdata.image_height
imw = nucdata.image_width
imc = nucdata.image_channels

class NucGaussConvNet:
  def __init__(self):
    self.saver = None
    self.x = tf.placeholder(tf.float32, [None, imh, imw, imc], name="input")
    self.y = tf.placeholder(tf.float32, [None, 1], name="labels")
    self.keep_prob = tf.placeholder(tf.float32)
    self.yPredition = None
    self.variables = []
    self.layers = []
    self.outs = {}
    self.init = None
    self.cross_entropy = None
    self.train = None
    self.accuracy = None
    self.sess = None

  def construct(self, nkernel):
    # construct the kernels
    kwidth = 101 # kernel width (kernel height should be the image height)
    kcenter = 50 # at the very center of the kernel
    kernels_multi_channel = []
    for i in xrange(nkernel):
      kernels_single_channel = []
      for polar in ["pos","neg"]:
        # the cluster sequence
        vname = "cluster_%d_%s" % (i,polar)
        clt = tf.Variable( tf.truncated_normal([imh,1],stddev=0.2) , name=vname)
        self.variables.append(clt)
        # convered gaussian variance
        vname = "gauss_var_%d_%s" % (i,polar)
        gv = tf.Variable( tf.truncated_normal([1],stddev=5)+pow(10,2) , name=vname)
        self.variables.append(gv)
        # total amplitude
        vname = "amp_%d_%s" % (i,polar)
        am = tf.Variable( tf.truncated_normal([1],stddev=0.05)+1 , name=vname)
        self.variables.append(am)
        # build the kernel
        vname = "gauss_curve_%d_%s" % (i,polar)
        seq = np.array(range(kwidth)).astype(np.float32).reshape([1,kwidth]) - kcenter
        gausscurve = tf.exp( -np.square(seq) / 2 / gv , name=vname )
        vname = "kernel_single_%d_%s" % (i,polar)
        kernel_single_channel = tf.matmul(clt,gausscurve,name=vname)
        kernels_single_channel.append(kernel_single_channel)
      vname = "kernel_multi_channel_%d" % i
      kernel_multi_channel = tf.stack(kernels_single_channel, axis=2, name=vname)
      kernels_multi_channel.append(kernel_multi_channel)
    vname = "kernel_conv"
    kernel = tf.stack(kernels_multi_channel, axis=3, name=vname)
    self.outs['kernel'] = kernel
    # pre-process the input data
    trimmed = self.x[:,:,200:-100,:]
    normed = trimmed / 50000.0
    # convolve
    vname = "conv"
    conved = tf.nn.conv2d(normed, kernel, (1,1,1,1), padding="VALID", name=vname)
    # activate
    vname = "conv_activated"
    activated = tf.sigmoid(conved, name=vname)
    # pooling
    poolsize = (1,1,10,1)
    vname = "pooling"
    pooling = tf.nn.max_pool(activated, poolsize, poolsize, padding="VALID", name=vname)
    # flatten
    yshape = pooling.shape.as_list()
    ndim_flatten = yshape[1]*yshape[2]*yshape[3]
    vname = "flatten"
    flatten = tf.reshape(pooling, [-1,ndim_flatten], name=vname)
    # dropout
    vname = "droped_out"
    dropedout = tf.nn.dropout(flatten, self.keep_prob, name=vname)
    # full connection
    weights = tf.Variable( tf.truncated_normal([ndim_flatten,1],stddev=0.2) , name=vname)
    bias = tf.Variable( tf.truncated_normal([1],stddev=0.2) , name=vname)
    self.variables.append(weights)
    self.variables.append(bias)
    vname = "y_out"
    y_out = tf.add(tf.matmul(dropedout,weights), bias, name=vname)
    self.yPredition = y_out

  def evaluate(self):
    # self.cross_entropy = tf.reduce_mean( -tf.reduce_sum( self.y*tf.log(self.yPredition) , 1 ) )
    self.cross_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.yPredition) )
    self.variables.append(self.cross_entropy)
    self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
    correct_prediction = tf.equal( tf.sigmoid(self.yPredition) > 0.5, self.y > 0.5 )
    self.accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

  def initialize(self, filename=None):
    self.saver = tf.train.Saver()
    # self.init = tf.variables_initializer(self.variables)
    self.init = tf.initialize_all_variables()
    self.sess = tf.Session()
    self.sess.run(self.init)
    if filename:
      self.saver.restore(self.sess, filename)

  def prepare(self, filename=None):
    self.construct(100)
    self.evaluate()
    self.initialize(filename)

  def train_step(self, batch_size=100, evaluate=False, loss=False):
    new_batch = dataset.train.next_batch(batch_size)
    batch_xs = new_batch.data
    batch_ys = new_batch.labels
    self.sess.run(self.train, feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.8})
    if evaluate:
      return self.sess.run([self.accuracy, self.cross_entropy], feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1})

  def test_step(self):
    return self.sess.run([self.accuracy, self.cross_entropy], feed_dict = {self.x: dataset.test.data, self.y: dataset.test.labels, self.keep_prob: 1})

  def outNames(self):
    return self.outs.keys()

  def readParam(self, paramName):
    val = self.sess.run(self.outs[paramName])
    return val

  def trainEpoches(self, epoches=1):
    batch_size = 100
    batch_num = dataset.train.data.shape[0] / batch_size
    for epoch in range(epoches):
      for i in range(batch_num):
        self.train_step()
    a = self.test_step()[0]

  def save(self, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self.saver.save(self.sess, filename)

def prepareNet(filename = None):
  net = NucGaussConvNet()
  net.prepare(filename)
  return net


save_dir = "../../train.ckpt"
save_filename = "save.ckpt"
save_path = os.path.join(save_dir,save_filename)
save_epoch_path = save_path + ".epoch"

if __name__ == '__main__':
  # fetch dataset
  dataset = nucdata.Nuc("../../datasets/TNT.bin")
  if len(sys.argv) == 1:
    tf.device("/gpu:1")
    batch_size = 100
    batch_num = dataset.train.data.shape[0] / batch_size
    net = NucGaussConvNet()
    start_epoch = 0
    if os.path.exists(save_dir):
      net.prepare(save_path)
      f = open(save_epoch_path)
      start_epoch = int(f.read())
      f.close()
      print "State loaded from: %s" % save_path
    else:
      net.prepare()
    for epoch in range(start_epoch, 1000):
      test = net.test_step()
      accuracy = test[0]
      loss = test[1]
      print "Epoch %d: accuracy = %f , loss = %f" % (epoch, accuracy, loss)
      if (epoch % 10) == 0:
        net.save(save_path)
        f = open(save_epoch_path, "w")
        f.write("%d"%epoch)
        f.close()
        print "State saved to: %s" % save_path
      for i in range(batch_num):
        net.train_step()
  elif sys.argv[1] == "w":
    if os.path.exists(save_dir):
      n = prepareNet(save_path)
      print "Accuracy: %f" % n.test_step()[0]
      print "Params:", n.outNames()
      kernel = n.readParam('kernel')
      import scipy.io as sio
      sio.savemat("/home/cosmo/downloads/w.mat", {'kernel': kernel})
    else:
      n = prepareNet()
      w0 = n.readParam('w1')
      print "Accuracy: %f" % n.trainEpoches(5)
      print "Params:", n.outNames()
      kernel = n.readParam('kernel')
      import scipy.io as sio
      sio.savemat("/home/cosmo/downloads/w.mat", {'kernel': kernel})