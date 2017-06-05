import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0] 
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      probs = np.exp(scores)/np.sum(np.exp(scores))
      loss += -np.log(probs[y[i]])
      probs[y[i]] -= 1
      for j in xrange(num_classes):
          dW[:,j] = X[i,:]*probs[j]
  #print "shape of loss",probs.shape          
  
  loss /= num_train
  dW /= num_train
  
  #print "Normal with Loops: "
  loss += 0.5 * reg * np.sum(W * W)
  #print "Loss: ", loss
  dW += reg * W  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # The number of training examples 
  num_train = X.shape[0]
  
  # computing the scores for each class on entire training set using the linear
  #  classifier F = Wx
  
  scores = X.dot(W) # NxD * DxC = NxC
    
  # Now subtract the max element of corresponding row from each row    
  scores -= np.max(scores,axis=1,keepdims=True)
  
  # Compute the softmax error on the entire set 
  probabilities = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
  
  # Only store the correct probabilities of a particular training example
  correct_probs = probabilities[range(num_train),y]
  
  # Compute the loss and also add the regularisation term using reg provided
  loss = np.sum(-np.log(correct_probs)) / num_train
  loss += 0.5 * reg * np.sum(W*W)    
  probabilities[range(num_train),y] -= 1

  # compute the gradient also before compiling verify dimensionality.
  dW = X.T.dot(probabilities) / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

  
  
     
  
