import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt
from operator import add
from functools import reduce
from sklearn.datasets import load_digits
import time
import numpy as np
from itertools import product
plt.rcParams["figure.figsize"] = (14,8)

from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from sklearn.cross_validation import train_test_split

def batch_generator(batch_size, data, labels=None):

    n_batches = int(np.ceil(data.size(0) / float(batch_size)))
    idx = var(torch.LongTensor(np.random.permutation(len(data))))
    
    data_shuffled = data[idx]
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield (data_shuffled[start:end, :].cuda(), labels[start:end].cuda())
        else:
            yield data_shuffled[start:end, :].cuda()

            
if torch.cuda.is_available():
    var = lambda x:Variable(x, volatile=True)
else:
    var = lambda x:Variable(x, volatile=True)

             
class BinaryConvolutionalRBM(nn.Module, BaseEstimator, TransformerMixin):

    def __init__(self, 
                 in_channels,
                 out_channels,
                 momentum,
                 kw=3,
                 learning_rate=1e-3,
                 n_epochs=200,
                 init_hidden_bias=-2,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True,
                 weight_decay=.01,
                 stride=1):
        super(BinaryConvolutionalRBM, self).__init__()
        self.kw = kw
        self.kh = kw
        self.nonlinearity = "sigmoid"
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.init_hidden_bias=init_hidden_bias
        self.weight_decay = weight_decay
        self.stride_n = stride
        self.fitted = False
    
    def fit(self, X):
        # Initialize RBM parameters
        if not self.fitted:
            self.kernel = nn.Parameter(torch.rand(self.out_channels, self.in_channels, self.kh, self.kw).cuda())
            nn.init.xavier_uniform(self.kernel, gain=nn.init.calculate_gain(self.nonlinearity))
            
            self.visible_shape = X.shape[1:]
            global i_hidden_bias
            hw = hh = self.visible_shape[1] - self.kw + 1
            
            self.h_bias = nn.Parameter(i_hidden_bias + torch.zeros(1, self.out_channels, 1, 1).cuda())
            self.v_bias = nn.Parameter(torch.rand(1, self.in_channels, 1, 1).cuda())

            self.visible_shape = X.shape[1:]
            
        if type(X) is np.ndarray:
            visible = var(torch.Tensor(X))
        else:
            visible = X

        self.batch_size = X.shape[0]
        self._stochastic_gradient_descent(visible)
        self.fitted = True
        
        return self

    def transform(self, X):

        if type(X) not in [Variable, torch.cuda.FloatTensor, torch.FloatTensor]:
            X = var(torch.Tensor(X))
            
        transformed_data = self._compute_hidden_units(X)
        return transformed_data

    def _reconstruct(self, transformed_data):

        return self._compute_visible_units(transformed_data)

    def _stochastic_gradient_descent(self, _data):

        #opt = torch.optim.SGD(self.parameters(), self.learning_rate, momentum=self.momentum)
        global _n_epochs
        for iteration in range(0, _n_epochs):
            idx = var(torch.LongTensor(np.random.permutation(len(_data))))
            data = _data[idx]

            for batch in batch_generator(self.batch_size, data):
                
                self._contrastive_divergence(batch)
                k = self.kernel.clone()
                
                for param in self.parameters():
                    param.data += self.learning_rate * param.grad.data
                    param.grad.data *= 0.
                
                
            if self.verbose or iteration == self.n_epochs-1:
                error = self._compute_reconstruction_error(data)
                print(">> Epoch %d finished \tRBM Reconstruction error %.9f" % (iteration+1, error))

    def _contrastive_divergence(self, vector_visible_units):
        
        v_0 = vector_visible_units
        v_t = var(v_0.data)

        # Sampling
        for t in range(self.contrastive_divergence_iter):
            h_t = self._sample_hidden_units(v_t)
            v_t = v_k = self._compute_visible_units(h_t)
        
        # Computing deltas
        h_0 = self._compute_hidden_units(v_0)
        h_k = self._compute_hidden_units(v_k)
        
        if self.v_bias.grad is None:
            self.v_bias.grad = var(torch.zeros(self.v_bias.size())).cuda()
        if self.h_bias.grad is None:
            self.h_bias.grad = var(torch.zeros(self.h_bias.size())).cuda()
        
        self.v_bias.grad += torch.mean(torch.mean(torch.mean(v_0 - v_k, 0, keepdim=True), 2, keepdim=True), 3, keepdim=True)
        self.h_bias.grad += torch.mean(torch.mean(torch.mean(h_0 - h_k, 0, keepdim=True), 2, keepdim=True), 3, keepdim=True)
        
        def calculate_gradient(v, h, kernel, scale = 1.):
            if kernel.grad is None:
                kernel.grad = var(torch.zeros(*kernel.size())).cuda()

            for i, j in product(range(kernel.size(0)), range(kernel.size(1))):
                v_, h_ = torch.transpose(v[:, j:j+1], 0, 1), torch.transpose(h[:, i:i+1], 0, 1).clone()                
                kernel.grad[i:i+1,j:j+1].data += scale * F.conv2d(v_, h_).data / v.size(0)

        calculate_gradient(v_0, h_0, self.kernel, scale=1.)
        calculate_gradient(v_k, h_k, self.kernel, scale=-1)

    def _sample_hidden_units(self, visible_units):

        hidden_units = self._compute_hidden_units(visible_units)
        return torch.bernoulli(hidden_units)

    def _sample_visible_units(self, hidden_units):
        visible_units = self._compute_visible_units(hidden_units)
        res = torch.bernoulli(visible_units)
        print("_sample_visible_units:", res)
        return res

    def _compute_hidden_units(self, visible_units):

        #eturn np.transpose(self._activation_function_class.function(
        #   np.dot(self.kernel, np.transpose(matrix_visible_units)) + self.h_bias[:, np.newaxis]))
        if type(visible_units.data) is torch.FloatTensor:
             hidden = torch.sigmoid(F.conv2d(visible_units, self.kernel.cpu(), stride=self.stride_n) + self.h_bias.cpu())
        else:
            hidden = torch.sigmoid(F.conv2d(visible_units, self.kernel, stride=self.stride_n) + self.h_bias)


        return hidden
        
    def _compute_visible_units(self, hidden_units):
        if type(hidden_units.data) is torch.FloatTensor:
            res = torch.sigmoid(F.conv_transpose2d(hidden_units, self.kernel.cpu(), stride=self.stride_n) + self.v_bias.cpu())
        else:
            res = torch.sigmoid(F.conv_transpose2d(hidden_units, self.kernel, stride=self.stride_n) + self.v_bias)
        return res

    def _compute_free_energy(self, visible_units):

        v = visible_units
        raise NotImplemented("_compute_free_energy")
        #return - np.dot(self.v_bias, v) - np.sum(np.log(1 + np.exp(np.dot(self.kernel, v) + self.h_bias)))
    
    def _compute_reconstruction_error(self, data):

        if type(data) is not Variable:
            data = Variable(data)
        data_transformed = self.transform(data)
        data_reconstructed = self._reconstruct(data_transformed)
        n = 30
        
        global i,ims2,print_every
        assert data_reconstructed.size(1) == self.in_channels, "wrong dimensions"
        
        if not i % print_every:
            #temp = np.concatenate(np.split(np.squeeze(data_reconstructed.data.cpu().numpy())[:n], np.arange(1, n)),2)[0]
            ims2.append(data_reconstructed.data.cpu().numpy())
            #if len(temp.shape) == 2:
            #    pass#plt.imshow(temp)
                #plt.show()
            #elif len(temp.shape) == 3:
                #print(temp.shape)
                
        i+=1
        return torch.mean(torch.sum(torch.sum(torch.sum((data_reconstructed - data) ** 2, -1), -1), -1))
    
    
def convert_to_image(results, n =10, skip=2):
    results = list(map(lambda x:np.squeeze(np.concatenate(np.split(np.squeeze(x)[:n], np.arange(1, n), 0), -1)), results[::skip]))
    resulting_image = np.vstack(results)
    plt.imshow(resulting_image)
    temp, plt.rcParams["figure.figsize"] = plt.rcParams["figure.figsize"],(20, 30)
    plt.show()
    plt.rcParams["figure.figsize"] = temp
    
# TODO: Debug this class, out of time
class ConvDBN(BaseEstimator, TransformerMixin):
    """
    This class implements a unsupervised Deep Belief Network.
    """

    def __init__(self,
                 structure_kernel_sizes=[3,3,3,3],
                 channels=[(1, 3), (3, 6), (6, 6)],
                 optimization_algorithm='sgd',
                 learning_rate_rbm=1e-3,
                 n_epochs_rbm=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 verbose=True,
                 weight_decay=.001):
        self.structure_kernel_sizes = structure_kernel_sizes
        self.channels = channels
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.rbm_class = BinaryConvolutionalRBM
        self.rbm_weight_decay = weight_decay

    def fit(self, X, y=None):
        """
        Fits a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        self.visible_size = X.shape[1:]
        
        # Initialize rbm layers
        self.rbm_layers = list()
        for kernel_width, (in_c, out_c) in zip(self.structure_kernel_sizes, self.channels):
            print(kernel_width, (in_c, out_c))
            rbm = BinaryConvolutionalRBM(in_c, out_c, .2, kernel_width,
                                 learning_rate=self.learning_rate_rbm,
                                 n_epochs=self.n_epochs_rbm,
                                 contrastive_divergence_iter=self.contrastive_divergence_iter,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose,
                                 weight_decay=self.rbm_weight_decay)
            
            self.rbm_layers.append(rbm)

        # Fit RBM
        if self.verbose:
            print("[START] Pre-training step:")
        input_data = X
        for rbm in self.rbm_layers:
            rbm.fit(input_data)
            rbm = rbm.cpu()
            input_data = rbm.transform(input_data)
            rbm.cuda()
        if self.verbose:
            print("[END] Pre-training step")
        return self
    
    def transform(self, X):
        """
        Transforms data using the fitted model.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        input_data = var(torch.Tensor(1.*X))
        for rbm in self.rbm_layers:
            input_data = rbm.transform(input_data)
        return input_data
    
    # TODO: Decide whether to sample from hidden units on way up
    def sample(self, n_samples=1, k=1, top_down=True):
        if top_down:
            n_top_features = self.rbm_layers[-1].visible_shape       
            hidden = 1*np.random.randint(0, 2, (n_samples, n_top_features[0], n_top_features[1], n_top_features[2]))
            hidden = var(torch.Tensor(hidden))
        else:
            hidden = self.transform(np.random.random((n_samples, self.visible_size[0], self.visible_size[1], self.visible_size[2])))
        

        hidden = self.rbm_layers[-1]._compute_visible_units(hidden)
        if len(self.rbm_layers) == 1:
            return hidden
        for i in range(k):
            net = self.rbm_layers[-1]._compute_hidden_units(hidden)
            activations = torch.bernoulli(net)
            
            hidden = torch.bernoulli(self.rbm_layers[-1]._compute_visible_units(activations))

        for rbm in reversed(self.rbm_layers[1:-1]):
            hidden = torch.bernoulli(rbm._compute_visible_units(hidden))
        
        return self.rbm_layers[0]._compute_visible_units(hidden)
    
    def plot_samples(self, n_samples=1, k=1, top_down=True):
        samples = self.sample(n_samples, k, top_down)
        n = n_samples        
        plt.imshow(np.squeeze(np.concatenate(np.split(np.squeeze(samples.data.cpu().numpy(), 1)[:n], np.arange(1, n)), 2)))
        plt.show()    
    
