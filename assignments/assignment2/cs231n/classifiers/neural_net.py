import numpy as np
import matplotlib.pyplot as plt
from layer_utils import *
from layers import *

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def init_three_layer_model(input_size,hidden_size,output_size,maxout=None,dropout=None):
  model = {}
  if maxout is not None:
    model['W1'] = 0.00001 * np.random.randn(maxout,input_size, hidden_size[0])
    model['b1'] = np.zeros((maxout,hidden_size[0]))
    model['W2'] = 0.00001 * np.random.randn(maxout,hidden_size[0],hidden_size[1])
    model['b2'] = np.zeros((maxout,hidden_size[1]))
  else:
    model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size[0])
    model['b1'] = np.zeros((hidden_size[0]))
    model['W2'] = 0.00001 * np.random.randn(hidden_size[0],hidden_size[1])
    model['b2'] = np.zeros((hidden_size[1]))

  model['W3'] = 0.00001 * np.random.randn(hidden_size[1], output_size)
  model['b3'] = np.zeros(output_size)
  return model
  

def Relu(x):
  return np.where(x>0,x,0)

def dRelu(x):
  return np.where(x>0,1,0)

def make_onehot(y,length):
  t = np.zeros((len(y),length))
  t[range(len(y)),y] = 1
  return t

def softmax(x):
  x = x.T - x.max(axis=1)
  x = np.exp(x)
  x = x/np.sum(x,axis=0)
  return x.T
  #x = x - x.max(axis=1)
  #x = np.exp(x)
  #x = x/np.sum(x,axis=1)
  #return x


  
def two_layer_net(X, model, y=None, reg=0.0, dropout=None,maxout=None,bn=None):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape
  scores = None
  a = Relu(X.dot(W1)+b1)
  scores = a.dot(W2)+b2 
  if y is None:
    return scores

  loss = None
  soft = softmax(scores)
  loss = np.sum(-np.log(soft[range(N),y]))/N + 0.5*reg*(np.sum(W1**2)+np.sum(W2**2))

  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  
  grads['W2'] = a.T.dot(soft - make_onehot(y,W2.shape[1]))/N + reg*W2
  grads['b2'] = np.sum(soft-make_onehot(y,W2.shape[1]),axis=0)/N 
  grads['W1'] = reg*W1 + X.T.dot((soft-make_onehot(y,W2.shape[1])).dot(W2.T)*dRelu(a))/N
  grads['b1'] = np.sum((soft-make_onehot(y,W2.shape[1])).dot(W2.T)*dRelu(a),axis=0)/N
  return loss, grads


def max_norm(param_name,param,C):
  """
  Although dropout alone gives significant improvements, 
  using dropout along with max- norm regularization, large decaying 
  learning rates and high momentum provides a significant 
  boost over just using dropout. 
  """
  norm = np.sqrt(np.sum(param*param))
  if norm > C:
    #print "param _ name : ",param_name 
    #print "norm of grads ", norm
    param = param/C
  return param


def two_layer_net1(X,model,y=None,reg=0.0,dropout=None,maxout=None,bn=None,C=2):
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N,D = X.shape
  scores = None
  
  dropout_param = {'p':dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'
  dropout_param['seed'] = 123

  if maxout is None:
    a1,cache1 = affine_relu_forward(X,W1,b1)
  else:
    a1,cache1 = maxout_forward(X,W1,b1)
    a2,cache3 = maxout_forward(a1,W2,b2)
  scores, cache4 = affine_forward(a1,W2,b2)
  if y is None:
    return scores

  data_loss, dscores = softmax_loss(scores,y)
  da1,dW2,db2 = affine_backward(dscores,cache4)
  
  if maxout is not None:
    da1,dW2,db2 = maxout_backward(da2,cache3)
    dx,dW1,db1 = maxout_backward(da1,cache1)
  else:
    dx,dW1,db1 = affine_relu_backward(da1,cache1)
    
  grads = {'W1':dW1, 'b1':db1,'W2':dW2, 'b2':db2}#,'W3':dW3, 'b3':db3}
  reg_loss = 0.0
  for p in ['W1','W2']:
    W = model[p]
    reg_loss += 0.5*reg*np.sum(W*W)
    grads[p] = max_norm(p,grads[p],C)
    grads[p] +=reg*W
  loss = data_loss + reg_loss
  return loss, grads
  
def three_layer_net_withbn(X,model,y=None,reg=0.0,dropout=None,maxout=None,bn=None,C=2):
  W1,b1,W2,b2,W3,b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'], model['b3']
  N,D = X.shape
  scores = None
  
  dropout_param = {'p':dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'
  dropout_param['seed'] = 123

  if maxout is None:
    a1,cache1 = affine_bn_relu_forward(X,W1,b1)
    d1,cache2 = dropout_forward(a1,dropout_param)
    a2,cache3 = affine_bn_relu_forward(d1,W2,b2)
    d2,cache3d = dropout_forward(a2,dropout_param)
  else:
    a1,cache1 = maxout_forward(X,W1,b1)
    a2,cache3 = maxout_forward(a1,W2,b2)
  scores, cache4 = affine_forward(d2,W3,b3)
  if y is None:
    return scores

  #print d1.shape, y.shape
  data_loss, dscores = softmax_loss(scores,y)
  da2,dW3,db3 = affine_backward(dscores,cache4)
  
  if maxout is not None:
    da1,dW2,db2 = maxout_backward(da2,cache3)
    dx,dW1,db1 = maxout_backward(da1,cache1)
  else:
    dd2 = dropout_backward(da2,cache3d)
    da1,dW2,db2 = affine_bn_relu_backward(dd2,cache3)
    dd1 = dropout_backward(da1,cache2)
    dx,dW1,db1 = affine_bn_relu_backward(dd1,cache1)
    
  grads = {'W1':dW1, 'b1':db1,'W2':dW2, 'b2':db2,'W3':dW3, 'b3':db3}
  reg_loss = 0.0
  for p in ['W1','W2','W3']:
    W = model[p]
    reg_loss += 0.5*reg*np.sum(W*W)
    grads[p] = max_norm(p,grads[p],C)
    grads[p] +=reg*W
  loss = data_loss + reg_loss
  return loss, grads


def three_layer_net(X,model,y=None,reg=0.0,dropout=None,maxout=None,bn=None,C=2):
  W1,b1,W2,b2,W3,b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'], model['b3']
  N,D = X.shape
  scores = None
  
  dropout_param = {'p':dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'
  dropout_param['seed'] = 123

  if maxout is None:
    a1,cache1 = affine_relu_forward(X,W1,b1)
    d1,cache2 = dropout_forward(a1,dropout_param)
    a2,cache3 = affine_relu_forward(d1,W2,b2)
    d2,cache3d = dropout_forward(a2,dropout_param)
  else:
    a1,cache1 = maxout_forward(X,W1,b1)
    a2,cache3 = maxout_forward(a1,W2,b2)
  scores, cache4 = affine_forward(d2,W3,b3)
  if y is None:
    return scores

  #print d1.shape, y.shape
  data_loss, dscores = softmax_loss(scores,y)
  da2,dW3,db3 = affine_backward(dscores,cache4)
  
  if maxout is not None:
    da1,dW2,db2 = maxout_backward(da2,cache3)
    dx,dW1,db1 = maxout_backward(da1,cache1)
  else:
    dd2 = dropout_backward(da2,cache3d)
    da1,dW2,db2 = affine_relu_backward(dd2,cache3)
    dd1 = dropout_backward(da1,cache2)
    dx,dW1,db1 = affine_relu_backward(dd1,cache1)
    
  grads = {'W1':dW1, 'b1':db1,'W2':dW2, 'b2':db2,'W3':dW3, 'b3':db3}
  reg_loss = 0.0
  for p in ['W1','W2','W3']:
    W = model[p]
    reg_loss += 0.5*reg*np.sum(W*W)
    grads[p] = max_norm(p,grads[p],C)
    grads[p] +=reg*W
  loss = data_loss + reg_loss
  return loss, grads


def three_layer_net1(X,model,y=None,reg=0.0,dropout=None,maxout=None,bn=None):
  W1,b1,W2,b2,W3,b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'], model['b3']
  N,D = X.shape
  scores = None
  
  dropout_param = {'p':dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'
  a1,cache1 = maxout_forward(X,W1,b1)
  d1,cache2 = dropout_forward(a1,dropout_param)
  a2,cache3 = maxout_forward(d1,W2,b2)
  scores, cache4 = affine_forward(a2,W3,b3)
  if y is None:
    return scores

  data_loss, dscores = softmax_loss(scores,y)
  da2,dW3,db3 = affine_backward(dscores,cache4)
  dd1,dW2,db2 = maxout_backward(da2,cache3)
  da1 = dropout_backward(dd1,cache2)
  dx,dW1,db1 = maxout_backward(da1,cache1)
  
  grads = {'W1':dW1, 'b1':db1,'W2':dW2, 'b2':db2,'W3':dW3, 'b3':db3}
  reg_loss = 0.0
  for p in ['W1','W2','W3']:
    W = model[p]
    reg_loss += 0.5*reg*np.sum(W*W)
    grads[p] +=reg*W
  loss = data_loss + reg_loss
  return loss, grads

