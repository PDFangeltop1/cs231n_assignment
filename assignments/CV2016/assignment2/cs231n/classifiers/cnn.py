import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import cPickle as pickle
import os    
import glob
import os.path as op
from cs231n.data_augmentation import *

def augment_fn(X, input_shape):
  out = random_flips(random_crops(X, input_shape[1:]))
  out = random_tint(random_contrast(out))
  return out

def predict_fn(X, input_shape):
  return fixed_crops(X, input_shape[1:], 'center')


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,loadData=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    assert filter_size % 2 == 1, 'Filter size must be odd; got %d'%filter_size
    self.params['W1'] = weight_scale*np.random.randn(num_filters, C, filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*H*W/4, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     


    
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    a1, cache1 = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    a2, cache2 = affine_relu_forward(a1,W2,b2)
    scores, cache3 = affine_forward(a2,W3,b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores,y)
    da2, dW3, db3 = affine_backward(dscores,cache3)
    da1, dW2, db2 = affine_relu_backward(da2,cache2)
    dx, dW1, db1 = conv_relu_pool_backward(da1,cache1)
    
    grads = {'W1':dW1, 'b1':db1, 'W2':dW2, 'b2':db2, 'W3':dW3, 'b3':db3}
    reg_loss = 0
    for idx in ['W1','W2','W3']:
      W = self.params[idx]
      grads[idx] += self.reg*W
      reg_loss += 0.5*self.reg*np.sum(W*W)
    loss = data_loss + reg_loss
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads
  
class MultiLayerConvNet(object):
  """
  A multilayer convolutional network with the following architecture:
  [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  [conv-relu-conv-relu-pool]*N - [affine-relu]*M, affine - [softmax or SVM]
  The network operators on minibatches of data that have shape (N,C,H,W)
  consisting of N images, each with height H and width W and with C input 
  channels
  """
  def __init__(self, input_dim=(3,32,32),num_filters=None,filter_size=5,
               hidden_dims=None, num_classes=10, weight_scale=1e-3, reg=0.1,
               dtype=np.float32,use_batchnorm=False,dropout=0,seed=None,
               loadData=None,predict_fn=None,augment_fn=None):

    """
    predic_fn, augment_fn: for data augmentation
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.num_filters = len(num_filters)
    self.filter_size = filter_size
    self.bn_batchnorm = use_batchnorm
    self.use_dropout = (dropout > 0)

    ############# the total number of layers including conv layer and affine layer#######
    self.num_layers = self.num_filters + len(hidden_dims) + 1
    #####################################################################################
#    print "how many layers ? ",self.num_layers

    self.bn_params = []
    self.dropout_param = {}
    
    self.predict_fn = predict_fn
    self.augment_fn = augment_fn
    if augment_fn is not None:
      input_dim = (3,28,28)
    
    self.input_dim = input_dim
    if loadData is not None:
      print "Load Data is ",loadData
      for f in glob.glob("%s/convNet_params_*.npy"%loadData):
        name_lst = op.splitext(op.basename(f))[0].split("_")
        if len(name_lst) == 3:
          param = name_lst[2]
          if param == "dropout":
            self.dropout_param = self.load_param(f)
          elif param == "bn":
            self.bn_params = self.load_param(f)
          else:
            self.params[param] = self.load_param(f) # W_i,b_i,beta_i, gamma_i
            print self.params[param].shape,
          print "load parameter %s successfully"%param
      return

    C,H,W = input_dim
    assert filter_size%2 == 1, 'Filter size must be odd: got %d'%filter_size
    all_filters = np.array([C])
    all_filters = np.concatenate((all_filters,np.array(num_filters)),axis=0)
    for i in range(self.num_filters):
      t = i + 1
      self.params['W%d'%t] = weight_scale*np.random.randn(all_filters[t],all_filters[t-1],filter_size,filter_size)
      self.params['b%d'%t] = np.zeros(all_filters[t])
      if self.bn_batchnorm is True:
        self.params['gamma%d'%t] = np.random.randn(all_filters[t])
        self.params['beta%d'%t] = np.random.randn(all_filters[t])
        
    all_hidden_layers = np.array([H*W*all_filters[-1]/np.power(4,self.num_filters)])
    a = np.array(hidden_dims)
    all_hidden_layers = np.concatenate((all_hidden_layers,a),axis=0)
    b = np.array([num_classes])
    all_hidden_layers = np.concatenate((all_hidden_layers,b))
    length = len(all_hidden_layers) - 1 
    for i in range(length):
      t = i + self.num_filters + 1
      self.params['W%d'%t] = weight_scale*np.random.randn(all_hidden_layers[i],all_hidden_layers[i+1])
      self.params['b%d'%t] = np.zeros(all_hidden_layers[i+1])
      if self.bn_batchnorm is True and i < length - 1:
        self.params['gamma%d'%t] = np.random.randn(all_hidden_layers[i+1])
        self.params['beta%d'%t] = np.random.randn(all_hidden_layers[i+1])
      
    if self.bn_batchnorm:
      self.bn_params = [{'mode':'train'} for i in xrange(self.num_layers)]

    if self.use_dropout is True:
      self.dropout_param = {'mode':'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    for k,v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
      
      
  def load_param(self,fname):
    with open(fname,"r")as fid:
      params = pickle.load(fid)
    return params

  def save_param(self,filename, params):
    with open (filename,"w") as f:
      pickle.dump(params,f)

  def save_params(self,filepath):
    for k,v in self.params.iteritems():
      filename = "%s/convNet_params_%s.npy"%(filepath,k)
      self.save_param(filename,v)
    
    filename = "%s/convNet_params_dropout.npy"%filepath
    self.save_param(filename,self.dropout_param)

    filename = "%s/convNet_params_bn.npy"%filepath
    self.save_param(filename,self.bn_params)
    
  def param_norm(self):
    x = 0
    for idx in self.params.keys():
      t = self.params[idx]
      x += np.sum(t)
    return x

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    if mode == 'test':
      if self.predict_fn is not None:
        X = predict_fn(X, self.input_dim)
    else:
      if self.augment_fn is not None:
        X = augment_fn(X, self.input_dim)
      
    if self.bn_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
     
    if self.use_dropout:
      self.dropout_param['mode'] = mode

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    caches = []
    activations = []
    activations.append(X)
    for i in range(self.num_layers):
      t = i + 1
#      print "i = ",i
      if i < self.num_filters:
        if self.bn_batchnorm is False:
          a, cache = conv_relu_pool_forward(activations[i],self.params['W%d'%t],
                                            self.params['b%d'%t],conv_param,pool_param)
        else:
#          print activations[i].shape
#          print 'W%d'%t, self.params['W%d'%t].shape, self.params['b%d'%t].shape

          a, cache = conv_batchnorm_relu_pool_forward(activations[i], self.params['W%d'%t],
                                                 self.params['b%d'%t],self.params['gamma%d'%t],
                                                 self.params['beta%d'%t],self.bn_params[i],
                                                 conv_param, pool_param)

      else:
        if self.bn_batchnorm is True and t < self.num_layers:
          a, cache = affine_batchnorm_dropout_forward(activations[i], self.params['W%d'%t],
                                                        self.params['b%d'%t],self.params['gamma%d'%t],
                                                        self.params['beta%d'%t],self.bn_params[i],
                                                        self.dropout_param)
        else:
          a, cache = affine_forward(activations[i],self.params['W%d'%t],self.params['b%d'%t])

      caches.append(cache)
      activations.append(a)

    scores = activations[-1]
    if mode == 'test':
      return scores

    loss, grads = 0, {}
    derivatives = []
    data_loss, dscores = softmax_loss(scores,y)
    reg_loss = 0
    derivatives.append(dscores)
    for i in range(self.num_layers):
      idx = self.num_layers - i 
      if idx > self.num_filters:
        if self.bn_batchnorm is True and i != 0:
          da, dw, db, dgamma, dbeta = affine_batchnorm_dropout_backward(derivatives[i],caches[idx-1])
        else:
          da, dw, db = affine_backward(derivatives[i],caches[idx-1])
      else:
        if self.bn_batchnorm is False:
          da, dw, db = conv_relu_pool_backward(derivatives[i],caches[idx-1])
        else:
          da, dw, db, dgamma, dbeta = conv_batchnorm_relu_pool_backward(derivatives[i],
                                                                        caches[idx-1])
      derivatives.append(da)
      if self.bn_batchnorm is True and i != 0:
        grads['gamma%d'%idx] = dgamma
        grads['beta%d'%idx] = dbeta
      W = self.params['W%d'%idx]
      grads['W%d'%idx] = dw + self.reg*W
      grads['b%d'%idx] = db 
      reg_loss += 0.5*self.reg*np.sum(W*W)
    loss = data_loss + reg_loss
    return loss, grads

def conv_batchnorm_relu_pool_forward(X, w, b,
                                     gamma, beta, bn_params,
                                     conv_param, pool_param):
  a, conv_cache = conv_forward_fast(X, w, b, conv_param)
  b, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_params)
  c, relu_cache = relu_forward(b)
  out , pool_cache = max_pool_forward_fast(c, pool_param)
  cache = (conv_cache, batch_cache, relu_cache, pool_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
  (conv_cache, batch_cache, relu_cache, pool_cache) = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dc = relu_backward(ds, relu_cache)
  db, dgamma, dbeta = spatial_batchnorm_backward(dc, batch_cache)
  dx, dw, db = conv_backward_fast(db,conv_cache)
  return dx, dw, db, dgamma, dbeta
  
def affine_batchnorm_dropout_forward(x,w,b,gamma,beta,bn_param,dropout_param):
  # print "x ",x.shape
  # print "w ",w.shape
  # print "b", b.shape
  # print "gamma: ",gamma.shape
  # print "beta: ",beta.shape
  #print "dropout ",dropout_param
  #print "bn_param ",bn_param

  a, fc_cache = affine_forward(x,w,b)
  b, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, dropout_cache = dropout_forward(b, dropout_param)
  cache = (fc_cache, batch_cache, dropout_cache)
  return out, cache

def affine_batchnorm_dropout_backward(dout, cache):
  (fc_cache, batch_cache, drop_cache) = cache
  dc = dropout_backward(dout, drop_cache)
  db1, dgamma, dbeta = batchnorm_backward(dc,batch_cache)
  dx, dw, db = affine_backward(db1, fc_cache)
  return dx, dw, db, dgamma, dbeta

pass

