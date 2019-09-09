import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. You may need to modify some of the                #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    # Calculo los puntajes
    scores = X[i].dot(W)
    scores -=  np.max(scores) #To avoid numerical issues

    # Aplico Softmax a los puntajes
    q = np.exp(scores) / np.sum(np.exp(scores))

    # Calculo el Loss de Entropīa Cruzada
    loss += -np.log(q[y[i]])

    # Calculo el Gradiente de W utilizando la regla de la cadena y el gradiente
    # de los puntajes (qi-yi).
    for j in xrange(num_classes):
      dW[:,j] += (q-(j == y[i]))[j] * X[i]

  # Divido entre la cantidad de elementos de entrenamiento
  loss /= num_train
  dW /= num_train
 
  # Agrego la regularización al loss
  loss += 0.5 * reg * np.sum(W * W)

  # Agrego la regularización al Gradiente
  dW += reg*W

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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Calculo los puntajes
  scores = X.dot(W)
  scores -=  np.max(scores) #To avoid numerical issues

  # Aplico Softmax a los puntajes
  q = np.exp(scores) / (np.sum(np.exp(scores), axis=1).reshape(num_train,1))

  # Calculo el Loss de Entropīa Cruzada
  loss_array = -np.log(q)

  # Tomo las etiquetas "y" de entrada asociados a cada elemento de X, y lo
  # transformo en una matriz donde cada fila es la representación one-hot para
  # el X respectivo.
  y_mat = np.zeros_like(loss_array)
  y_mat[range(num_train), y] = 1

  # Me quedo sólo con los loss correctos, es decir la diferencia con las
  # clasificaciones correctas
  correct_loss_array = y_mat * loss_array

  # Hago la sumatoria
  loss = np.sum(correct_loss_array)
  
  # Divido entre la cantidad de elementos de entrenamiento
  loss /= num_train

  # Agrego la regularización al loss
  loss += 0.5 * reg * np.sum(W * W)

  # Calculo el Gradiente de W utilizando la regla de la cadena y el gradiente
  # de los puntajes (qi-yi).
  dW = q
  dW = q-y_mat
  dW = np.dot(X.T, dW)
  dW /= num_train

  # Agrego la regularización al Gradiente
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

