import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

####################################

class ReLULayer(object):
    def forward(self, input_):
        # remember the input for later backpropagation
        self.input = input_
        # return the ReLU of the input
        relu = np.maximum(0, input_) #compute relu
        return relu

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        """
        ReLU'(t) is 0 for t < 0, else 1 
        """
        downstream_gradient = upstream_gradient * (self.input > 0).astype(int) 
        return downstream_gradient

    def update(self, learning_rate):
        pass # ReLU is parameter-free

####################################

class OutputLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input_):
        # remember the input for later backpropagation
        self.input = input_
        # return the softmax of the input
        softmax = np.exp(input_) / np.sum(np.exp(input_), axis=1)[:,None] # compute softmax for every instance, 
        # add dimension by using index None which has been lost by contracting over axis = 1
        return softmax

    def backward(self, predicted_posteriors, true_labels):
        # return the loss derivative with respect to the stored inputs
        # (use cross-entropy loss and the chain rule for softmax,
        #  as derived in the lecture)
        """
        Compute derivative of Cross-Entropy according to lecture, 
        delta_tilde_{lk} = Z_{lk} - 1 if y = k, Z_{lk} else
        y corresponds to true_labels, Z_{lk} to predicted_posteriors
        """
        downstream_gradient = np.array([predicted_posteriors[:,0] - (true_labels==0).astype(int),\
                                              predicted_posteriors[:,1] - (true_labels==1).astype(int)]).T
        return downstream_gradient

    def update(self, learning_rate):
        pass # softmax is parameter-free

####################################

class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        # randomly initialize weights and intercepts
        # initialize according to lecture, note the factor 2 for ReLU
        self.B = np.random.normal(loc=0, scale=2/n_inputs, size=(n_outputs, n_inputs))
        self.b = np.random.normal(loc=0, scale=2/n_inputs, size=n_outputs) 

    def forward(self, input_):
        # remember the input for later backpropagation
        self.input = input_
        # compute the scalar product of input and weights
        # (these are the preactivations for the subsequent non-linear layer)
        preactivations = np.dot(self.B, input_.T).T + self.b #linear mapping
        return preactivations

    def backward(self, upstream_gradient):
        # compute the derivative of the weights from
        # upstream_gradient and the stored input
        self.grad_b = np.sum(upstream_gradient, axis=0) #sum over N instances
        # compute the outer product of each row of upstream_gradient with the corresponding row
        # of input yielding N matrices according to delta_B = delta_tilde^T * Z^T ; then sum over all N matrices
        self.grad_B = np.sum(np.einsum('ij,ik->ijk', upstream_gradient, self.input),axis=0)
        # compute the downstream gradient to be passed to the preceding layer
        downstream_gradient = np.dot(upstream_gradient, self.B) # your code here
        return downstream_gradient

    def update(self, learning_rate):
        # update the weights by batch gradient descent
        self.B = self.B - learning_rate * self.grad_B
        self.b = self.b - learning_rate * self.grad_b

####################################

class MLP(object):
    def __init__(self, n_features, layer_sizes):
        # constuct a multi-layer perceptron
        # with ReLU activation in the hidden layers and softmax output
        # (i.e. it predicts the posterior probability of a classification problem)
        #
        # n_features: number of inputs
        # len(layer_size): number of layers
        # layer_size[k]: number of neurons in layer k
        # (specifically: layer_sizes[-1] is the number of classes)
        self.n_layers = len(layer_sizes)
        self.layers   = []

        # create interior layers (linear + ReLU)
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out))
            self.layers.append(ReLULayer())
            n_in = n_out

        # create last linear layer + output layer
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        self.layers.append(OutputLayer(n_out))

    def forward(self, X):
        # X is a mini-batch of instances
        batch_size = X.shape[0]
        # flatten the other dimensions of X (in case instances are images)
        X = X.reshape(batch_size, -1)

        # compute the forward pass
        # (implicitly stores internal activations for later backpropagation)
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, predicted_posteriors, true_classes):
        # perform backpropagation w.r.t. the prediction for the latest mini-batch X
        # start backpropagation in the output layer
        layer = self.layers[-1]
        downstream_gradient = layer.backward(predicted_posteriors, true_classes)
        for layer in self.layers[:-1][::-1]: # propagate gradient to layler l, loop backwards through layers(except for last layer)
            downstream_gradient = layer.backward(downstream_gradient)

    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        self.backward(posteriors, Y)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate, x_test, y_test):
        N = len(x)
        n_batches = N // batch_size
        train_errors = np.zeros(n_epochs)
        test_errors  = np.zeros_like(train_errors)
        for i in range(n_epochs):
            # print("Epoch", i)
            # reorder data for every epoch
            # (i.e. sample mini-batches without replacement)
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # create mini-batch
                start = batch * batch_size
                x_batch = x[permutation[start:start+batch_size]]
                y_batch = y[permutation[start:start+batch_size]]

                # perform one forward and backward pass and update network parameters
                self.update(x_batch, y_batch, learning_rate)

            # report training loss every epoch            
            predicted_posteriors = self.forward(x)
            # determine class predictions from posteriors by winner-takes-all rule
            predicted_classes = np.argmax(predicted_posteriors, axis=1) 
            # compute and output the error rate of predicted_classes
            error_rate = np.sum(predicted_classes != y) / len(y) 
            train_errors[i] = error_rate
            
            # do the same for test set
            predicted_posteriors = self.forward(x_test)
            predicted_classes = np.argmax(predicted_posteriors, axis=1) 
            error_rate = np.sum(predicted_classes != y_test) / len(y_test) 
            test_errors[i] = error_rate
            
        return train_errors, test_errors
    
##################################
"""
utility functions
"""        
    
def visualize_data(X, Y, title = None, save = False):
    """
    visualizes classes Y of given 2D-dataset X
    """
    plt.figure(figsize = (8,8))
    plt.scatter(X[:,0], X[:,1], c = Y)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    if title:
        plt.title(title)
    if save:
        plt.savefig(title + '.png', dpi = 300)

def plot_error_rates(x1, x2, title = None, save = False, label1 = 'training error', label2 = 'validation error'):
    """
    plots the error rate for every epoch of training, 
    x holding the respective rates (hence len(x) = n_epochs)
    """
    plt.figure(figsize = (9,7))
    plt.plot(x1, label = label1, lw = 2.5)
    #plt.plot(x2, label = label2, lw = 2.5)
    plt.xlabel('epoch $n$')
    plt.ylabel('error rate')
    plt.ylim(0,1)
    plt.legend(loc = 'best')
    if title:
        plt.title(title)
    if save:
        plt.savefig(title + '.png', dpi = 300)

def run_ensemble(N, n_features, n_classes, layer_sizes, n_epochs, batch_size, lr, n_ensemble):
    """
    repeat a classification task n_ensemble times.
    """
    train_ens = np.zeros((n_ensemble, n_epochs))
    for k in range(n_ensemble):
        X_train, Y_train = datasets.make_moons(N, noise=0.05)
        X_test,  Y_test  = datasets.make_moons(N, noise=0.05)
        offset  = X_train.min(axis=0)
        scaling = X_train.max(axis=0) - offset
        X_train = ((X_train - offset) / scaling - 0.5) * 2.0
        X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0
        network = MLP(n_features, layer_sizes)
        train_errors, test_errors = network.train(X_train, Y_train, n_epochs, batch_size, learning_rate, X_test, Y_test)
        train_ens[k,:] = train_errors
    return train_ens

def plot_ensemble(ensemble, n_std, save = False, title = None):
    """
    plot mean and standard deviation of a given set of error rates.
    """
    # define some custom colors  #orange, blue, green, red
    plt.figure(figsize = (9,7))
    plt.title('Mean and median out of ' + str(len(ensemble)) + ' runs; showing ' + str(n_std) + ' standard deviations (shaded)')
    colors = np.array([[0.9804, 0.4902, 0], [0.24, 0.45, 0.8223], [0.5490, 0.6765, 0.1912], [0.816, 0.044, 0.008]])    
    std = np.std(ensemble, axis = 0)
    mean = np.mean(ensemble, axis = 0)
    medi = np.median(ensemble, axis = 0)
    
    s_up = mean + n_std * std
    s_lo = mean - n_std * std
    plt.plot(s_up, color = colors[1], lw = 0.5, alpha = 0.15)
    plt.plot(s_lo, color = colors[1], lw = 0.5, alpha = 0.15)
    plt.fill_between(np.arange(len(mean)), mean, s_up, color = colors[1], alpha = 0.1)
    plt.fill_between(np.arange(len(mean)), mean, s_lo, color = colors[1], alpha = 0.1)
    plt.plot(medi, color = colors[3], lw = 2.0, label = 'median')
    plt.plot(mean, color = colors[1], lw = 2.0, label = 'mean')
    plt.legend(loc = 'best')
    if save:
        if title:
            plt.savefig(title + '.png', dpi = 300)

#########################################
if __name__=="__main__":

    # set training/test set size
    N = 2000

    # create training and test data
    X_train, Y_train = datasets.make_moons(N, noise=0.05)
    X_test,  Y_test  = datasets.make_moons(N, noise=0.05)
    
    # alternatively, use
    #X_train, Y_train = datasets.make_cricles(N, factor=0.5, noise=0.05)
    #X_test,  Y_test  = datasets.make_circles(N, factor=0.5, noise=0.05)
    
    n_features = 2
    n_classes  = 2

    # standardize features to be in [-1, 1]
    offset  = X_train.min(axis=0)
    scaling = X_train.max(axis=0) - offset
    X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0

    # set hyperparameters (play with these!)
    # 10,10, 1000,200,0.0001 work well
    # also 5, 5, 200, 200, 2e-4
    layer_sizes = [10,10, n_classes]
    n_epochs = 400
    batch_size = 200
    learning_rate = 0.0001

    # create network
    network = MLP(n_features, layer_sizes)

    # train and compute error rates for each epoch (for test and train set each)
    train_errors, test_errors = network.train(X_train, Y_train, n_epochs, batch_size, learning_rate, X_test, Y_test)

    # test
    predicted_posteriors = network.forward(X_test)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = np.argmax(predicted_posteriors, axis=1) 
    # compute and output the error rate of predicted_classes
    error_rate = np.sum(predicted_classes != Y_test) / len(Y_test) 
            
    # visualize data
    visualize_data(X_train, Y_train, title = 'Training Set')
    visualize_data(X_test, Y_test, title = 'Test Set') 
    visualize_data(X_test, predicted_classes, title = 'Prediction', save = True)
    # plot train/val error
    #plot_error_rates(train_errors, test_errors)
    ens = run_ensemble(N, n_features, n_classes, layer_sizes, n_epochs, batch_size, learning_rate, 10)
    plot_ensemble(ens, 3)
    
    print("error rate for test set:", error_rate)

