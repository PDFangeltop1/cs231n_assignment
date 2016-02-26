import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]

    count = 0 # how many cases bigger than 1
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count += 1
        dW[j] += X[:,i]
        loss += margin
    dW[y[i]] += X[:,i]*count*(-1)

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)
  N = len(X[0])
  correct_class_scores = scores[y,np.arange(N)]
  margins = scores - correct_class_scores
  margins = np.where(margins !=0, margins+1,0)
  margins = np.where(margins<0, 0, margins)
  loss = np.sum(margins)/N
  loss += 0.5*reg*np.sum(W*W)

  margins = np.where(margins>0,1,0)
  col_sum = np.sum(margins, axis=0) # N-dim vector 
  margins[y,range(N)] = -col_sum[range(N)]
  dW += margins.dot(X.T)
  dW /= N
  dW += reg*W

  return loss, dW
