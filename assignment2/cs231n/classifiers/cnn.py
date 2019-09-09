from builtins import object
import numpy as np
import math

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network w_ith the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        self.params['W1'] = np.random.normal(0,weight_scale,[num_filters, input_dim[0], filter_size, filter_size])
        self.params['b1'] = np.zeros(num_filters)
        
        self.params['W2'] = np.random.normal(0,weight_scale,[int(num_filters*input_dim[1]*input_dim[2]/4), hidden_dim])
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim,num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
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
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        reg = self.reg

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        out2, cache2 = affine_relu_forward(out1,W2,b2)

        scores, cache3 = affine_forward(out2,W3,b3)
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
        loss, dL = softmax_loss(scores, y)

        # Agrego la regularización al loss
        loss += 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)

        dx3, grads['W3'], grads['b3'] = affine_backward(dL, cache3)
        dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3,cache2)
        _  , grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2,cache1)

        # Agrego regularización a los gradientes
        grads['W3'] += reg * W3
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedConvNet(object):
    """
    A fully-connected convolutional network with the following architecture:

    [conv-batchnorm-relu-pool]XN - [affine]XM - [softmax]

    where pool are 2x2 max pools
    """

    def __init__(self, conv_params, affine_hidden_dims, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, dropout=0.0, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - conv_params: following params for conv layers
            - num_filters: List with the number of filters to use in each convolutional layer
            - filter_size: List with the size of filters to use in each convolutional layer
        - affine_hidden_dims: List with the number of units to use in each fully-connected hidden layers
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        

        self.params = {}
        self.reg    = reg
        self.dtype  = dtype
        self.num_conv_layers   = len(conv_params['num_filters'])
        self.num_affine_layers = len(affine_hidden_dims)
        self.conv_params       = conv_params

        conv_filter_number = conv_params['num_filters']
        conv_filter_size   = conv_params['filter_size']

        ############################################################################
        # TODO: Initialize weights and biases for the fully-connected convolutional#
        # network.                                                                 #
        ############################################################################
        for conv_layer in range(0, self.num_conv_layers):
            
            # Usamos índices a partir de num_affine_layers porque los primeros
            # num_affine_layers los van a usar las capas fully connected
            index = conv_layer+1+self.num_affine_layers
            
            W_i  = 'W'+str(index)
            b_i  = 'b'+str(index)
            gamma_i = 'gamma'+str(index)
            beta_i  = 'beta'+str(index)

            if conv_layer == 0:
                self.params[W_i] = np.random.normal(0, weight_scale,[conv_filter_number[conv_layer],input_dim[0], conv_filter_size[conv_layer], conv_filter_size[conv_layer]])
            else:
                self.params[W_i] = np.random.normal(0, weight_scale,[conv_filter_number[conv_layer],conv_filter_number[conv_layer-1],conv_filter_size[conv_layer], conv_filter_size[conv_layer]])

            self.params[b_i]     = np.zeros(conv_filter_number[conv_layer])
            self.params[gamma_i] = np.ones( conv_filter_number[conv_layer])
            self.params[beta_i]  = np.zeros(conv_filter_number[conv_layer])

        #########
        affine_input_dim  = conv_filter_number[-1]*input_dim[1]*input_dim[2]
        affine_input_dim /= math.pow(2, 2*len(conv_filter_number))

        self.FullyConnectedNet = FullyConnectedNet(affine_hidden_dims, affine_input_dim, num_classes, dropout, True, 1e-4, weight_scale)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in range(self.num_conv_layers)]
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        ############## 3_CNN ############## CHEQUEAR
        # pass conv_param to the forward pass for the convolutional layer
        # filter_size = W1.shape[2]
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2} #####


        ################ FINAL ################
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        reg = self.reg
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected convolutional   #
        # net, computing the class scores for X and storing them in the scores     #
        # variable.                                                                #
        ############################################################################
        input_conv = X
        cache = {}

        for conv_layer in range(self.num_conv_layers):
            # Claridad
            index = conv_layer+1+self.num_affine_layers
            W_i = self.params['W'+str(index)]
            b_i = self.params['b'+str(index)]
            gamma_i = self.params['gamma'+str(index)]
            beta_i  = self.params['beta'+str(index)]
            # conv_params = self.conv_params[conv_layer] ########
            conv_param = {'stride': 1, 'pad': (self.conv_params['filter_size'][conv_layer] - 1) // 2}
            bn_params   = self.bn_params[conv_layer]

            # Hago el forward pass para cada capa de convolución
            input_conv, cache[index] = conv_batchnorm_relu_pool_forward(input_conv, W_i, b_i, gamma_i, beta_i, conv_param, bn_params, pool_param)

        # Genero la entrada de la FullyConnectedLayer a partir de la salida de la última capa
        # de convolución, guardo el shape de la salida de la última capa para usar en el backward
        input_conv_shape = input_conv.shape
        input_fully_connected = input_conv.reshape(input_conv_shape[0], np.prod(input_conv_shape)/input_conv_shape[0])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # Si es test devuelvo los cores y termino, si no sigo y calculo el loss y el gradiente
        if y is None:
            scores = self.FullyConnectedNet.loss(input_fully_connected, y)
            return scores
        loss, grads = self.FullyConnectedNet.loss(input_fully_connected, y)
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected w_ith L2 reg.   #
        ############################################################################

        # Agrego la regularización al loss
        for i in range(1,self.num_conv_layers+1):
            W_i = self.params['W'+str(i+self.num_affine_layers)]
            loss += 0.5*reg*np.sum(W_i*W_i)

        # Obtengo el gradiente según la entrada de la FullyConnected
        dx = grads.pop('dInput').reshape(input_conv_shape)

        for conv_layer in range(self.num_conv_layers,0,-1):
            index = conv_layer+self.num_affine_layers
            W_i = 'W'+str(index)
            b_i = 'b'+str(index)
            gamma_i = 'gamma'+str(index)
            beta_i  = 'beta'+str(index)

            dx, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = conv_batchnorm_relu_pool_backward(dx, cache[index]) 
        
        # Agrego regularización a los gradientes
        for i in range(1,self.num_conv_layers+1):
            W_i = self.params['W'+str(i+self.num_affine_layers)]
            grads['W'+str(i+self.num_affine_layers)] += reg * W_i
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedConvNet2(object):
    """
    A fully-connected convolutional network with the following architecture:

    [conv-bn-relu-conv-bn-relu-pool]xN - [affine]xM - [softmax]

    where pool are 2x2 max pools
    """

    def __init__(self, conv_params, affine_hidden_dims, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, dropout=0.0, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - conv_params: following params for conv layers
            - num_filters: List with the number of filters to use in each convolutional layer
            - filter_size: List with the size of filters to use in each convolutional layer
        - affine_hidden_dims: List with the number of units to use in each fully-connected hidden layers
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        

        self.params = {}
        self.reg    = reg
        self.dtype  = dtype
        self.num_conv_layers   = len(conv_params['num_filters'])
        self.num_affine_layers = len(affine_hidden_dims)
        self.conv_params       = conv_params

        conv_filter_number = conv_params['num_filters']
        conv_filter_size   = conv_params['filter_size']

        ############################################################################
        # TODO: Initialize weights and biases for the fully-connected convolutional#
        # network.                                                                 #
        ############################################################################
        capas_internas = 0
        for conv_layer in range(0, self.num_conv_layers):
            
            # Usamos índices a partir de num_affine_layers porque los primeros
            # num_affine_layers los van a usar las capas fully connected,
            # se suma la cantidad de capas internas para corregir el index
            index = conv_layer+1+self.num_affine_layers + capas_internas
            
            # Primera Capa Interna
            W_i  = 'W'+str(index)
            b_i  = 'b'+str(index)
            gamma_i = 'gamma'+str(index)
            beta_i  = 'beta'+str(index)

            if conv_layer == 0:
                self.params[W_i] = np.random.normal(0, weight_scale,[conv_filter_number[conv_layer],
                                                                     input_dim[0], 
                                                                     conv_filter_size[conv_layer], 
                                                                     conv_filter_size[conv_layer]])
            else:
                self.params[W_i] = np.random.normal(0, weight_scale,[conv_filter_number[conv_layer],
                                                                     conv_filter_number[conv_layer-1],
                                                                     conv_filter_size[conv_layer],
                                                                     conv_filter_size[conv_layer]])

            self.params[b_i]     = np.zeros(conv_filter_number[conv_layer])
            self.params[gamma_i] = np.ones( conv_filter_number[conv_layer])
            self.params[beta_i]  = np.zeros(conv_filter_number[conv_layer])

            # Segunda Capa interna
            W_i  = 'W'+str(index+1)
            b_i  = 'b'+str(index+1)
            gamma_i = 'gamma'+str(index+1)
            beta_i  = 'beta'+str(index+1)

            
            self.params[W_i] = np.random.normal(0, weight_scale,[conv_filter_number[conv_layer],
                                                                 conv_filter_number[conv_layer],
                                                                 conv_filter_size[conv_layer],
                                                                 conv_filter_size[conv_layer]])

            self.params[b_i]     = np.zeros(conv_filter_number[conv_layer])
            self.params[gamma_i] = np.ones( conv_filter_number[conv_layer])
            self.params[beta_i]  = np.zeros(conv_filter_number[conv_layer])

            # Acumulador Capas Internas
            capas_internas += 1



        #########
        affine_input_dim  = conv_filter_number[-1]*input_dim[1]*input_dim[2]
        affine_input_dim /= math.pow(2, 2*len(conv_filter_number))

        self.FullyConnectedNet = FullyConnectedNet(affine_hidden_dims, affine_input_dim, num_classes, dropout, True, reg, weight_scale)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in range(self.num_conv_layers*2)]
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        ############## 3_CNN ############## CHEQUEAR
        # pass conv_param to the forward pass for the convolutional layer
        # filter_size = W1.shape[2]
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2} #####


        ################ FINAL ################
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        reg = self.reg
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected convolutional   #
        # net, computing the class scores for X and storing them in the scores     #
        # variable.                                                                #
        ############################################################################
        input_conv = X
        cache = {}

        capas_internas = 0
        for conv_layer in range(self.num_conv_layers):
            # Claridad
            index = conv_layer+1+self.num_affine_layers+capas_internas

            W_i = self.params['W'+str(index)]
            b_i = self.params['b'+str(index)]
            gamma_i = self.params['gamma'+str(index)]
            beta_i  = self.params['beta'+str(index)]
            # conv_params = self.conv_params[conv_layer] ########
            conv_param = {'stride': 1, 'pad': (self.conv_params['filter_size'][conv_layer] - 1) // 2}
            bn_param_i = self.bn_params[conv_layer+capas_internas]

            W_i_2 = self.params['W'+str(index+1)]
            b_i_2 = self.params['b'+str(index+1)]
            gamma_i_2 = self.params['gamma'+str(index+1)]
            beta_i_2  = self.params['beta'+str(index+1)]
            bn_param_i_2 = self.bn_params[conv_layer+capas_internas+1]

            capas_internas += 1

            # Hago el forward pass para cada capa de convolución
            input_conv, cache[index] = conv_batchnorm_relu_conv_batchnorm_relu_pool_forward(input_conv,
                                                                                            W_i,   b_i  , gamma_i,   beta_i,   conv_param, bn_param_i,
                                                                                            W_i_2, b_i_2, gamma_i_2, beta_i_2, conv_param, bn_param_i_2,
                                                                                            pool_param)

        # Genero la entrada de la FullyConnectedLayer a partir de la salida de la última capa
        # de convolución, guardo el shape de la salida de la última capa para usar en el backward
        input_conv_shape = input_conv.shape
        input_fully_connected = input_conv.reshape(input_conv_shape[0], np.prod(input_conv_shape)/input_conv_shape[0])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # Si es test devuelvo los cores y termino, si no sigo y calculo el loss y el gradiente
        if y is None:
            scores = self.FullyConnectedNet.loss(input_fully_connected, y)
            return scores
        loss, grads = self.FullyConnectedNet.loss(input_fully_connected, y)
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected w_ith L2 reg.   #
        ############################################################################

        # Agrego la regularización al loss
        capas_internas = 0
        for i in range(1,self.num_conv_layers+1):
            index = i+self.num_affine_layers+capas_internas
            
            # Primera Capa Interna
            W_i = self.params['W'+str(index)]
            loss += 0.5*reg*np.sum(W_i*W_i)
            
            # Segunda Capa Interna
            W_i = self.params['W'+str(index+1)]
            loss += 0.5*reg*np.sum(W_i*W_i)

            # Acumulador Capas Internas
            capas_internas += 1

        # Obtengo el gradiente según la entrada de la FullyConnected
        dx = grads.pop('dInput').reshape(input_conv_shape)

        for conv_layer in range(self.num_conv_layers,0,-1):
            index = (conv_layer*2)+self.num_affine_layers
            
            W_i = 'W'+str(index-1)
            b_i = 'b'+str(index-1)
            gamma_i = 'gamma'+str(index-1)
            beta_i  = 'beta'+str(index-1)

            W_i_2 = 'W'+str(index)
            b_i_2 = 'b'+str(index)
            gamma_i_2 = 'gamma'+str(index)
            beta_i_2  = 'beta'+str(index)

            dx, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i], grads[W_i_2], grads[b_i_2], grads[gamma_i_2], grads[beta_i_2] = conv_batchnorm_relu_conv_batchnorm_relu_pool_backward(dx, cache[index-self.num_affine_layers]) 
        
        # Agrego regularización a los gradientes
        capas_internas = 0
        for i in range(1,self.num_conv_layers+1):

            index = i+self.num_affine_layers+capas_internas

            # Primera Capa Interna
            W_i = self.params['W'+str(index)]
            grads['W'+str(index)] += reg * W_i

            # Segunda Capa Interna
            W_i = self.params['W'+str(index+1)]
            grads['W'+str(index+1)] += reg * W_i

            # Acumulador Capas Internas
            capas_internas += 1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


# Utils Arquitectura 1
def conv_batchnorm_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    an_relu, relu_cache = relu_forward(an)
    out, pool_cache = max_pool_forward_fast(an_relu, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    dan = relu_backward(ds, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

# Utils Arquitectura 2
def conv_batchnorm_relu_conv_batchnorm_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, w2, b2, gamma2, beta2, conv_param_2, bn_param_2, pool_param):
    a, conv1_cache = conv_forward_fast(x, w, b, conv_param)
    a_bn, bn1_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    a_bn_relu, relu1_cache = relu_forward(a_bn)
    a_bn_relu_conv, conv2_cache = conv_forward_fast(a_bn_relu, w2, b2, conv_param_2)
    a_bn_relu_conv_bn, bn2_cache = spatial_batchnorm_forward(a_bn_relu_conv, gamma2, beta2, bn_param_2)
    a_bn_relu_conv_bn_relu, relu2_cache = relu_forward(a_bn_relu_conv_bn)

    out, pool_cache = max_pool_forward_fast(a_bn_relu_conv_bn_relu, pool_param)
    cache = (conv1_cache, bn1_cache, relu1_cache, conv2_cache, bn2_cache, relu2_cache, pool_cache)
    return out, cache

def conv_batchnorm_relu_conv_batchnorm_relu_pool_backward(dout, cache):
    conv1_cache, bn1_cache, relu1_cache, conv2_cache, bn2_cache, relu2_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    dan2 = relu_backward(ds, relu2_cache)
    da2, dgamma2, dbeta2 = spatial_batchnorm_backward(dan2, bn2_cache)
    dx2, dw2, db2 = conv_backward_fast(da2, conv2_cache)
    dan = relu_backward(dx2, relu1_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan,bn1_cache)
    dx, dw, db = conv_backward_fast(da, conv1_cache)
    return dx, dw, db, dgamma, dbeta, dw2, db2, dgamma2, dbeta2



