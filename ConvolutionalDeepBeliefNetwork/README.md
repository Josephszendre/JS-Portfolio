The Deep Belief Network (Hinton, 2006) is the first deep neural network to be successfully trained using backpropogation. This is remarkable as it was the first deep learning model that actually worked, 6 years before the deep learning wave took off. 

A deep belief network is trained using greedy layer-wise training of a stack of restricted boltzmann machines. Greedy layer-wise training is done by training each layer independent of the others in succession. After a layer is trained, the data is mapped through it and is used to train the next layer.

Restricted Boltzmann Machines are a stochastic neural network that is trained using an algorithm called Contrastive Divergence. Binary RBMs take binary inputs and return binary outputs. The binary activations for both visible and hidden are obtained by bernoulli sampling of the sigmoid function applied to the net (floating point) activations.

The differentiator in this model is that it uses the same sparse connectivity and weight sharing as convolutional neural networks.

My model learns a generative model of MNIST in minutes.

