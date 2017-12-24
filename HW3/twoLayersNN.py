import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        pass
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        # 2 NN structure =  input layer: x -> hidden layer: hidden -> output layer: scores

        w1= self.params['w1']
        b1=self.params['b1']
        w2= self.params['w2']
        b2=self.params['b2']


        # forward pass

        z1 = np.dot(x, w1) + b1 # intermediate value for hidden layer calculation
        a1=np.maximum(z1,0.01*z1) # hidden layer calculation with leaky relu neuron
        scores = a1.dot(w2) + b2  # score calculation
        scoresF= np.maximum(scores,.01*scores)

        # softmax loss calculation

        tmp = -np.log(np.exp(scoresF[range(len(y)), y]) / np.sum(np.exp(scoresF), axis=1)) #loss
        l2= reg * (np.sum(w1 ** 2) +np.sum(w2 ** 2)) # l2 regularization
        loss = np.sum(tmp) / len(y) + l2 # NN loss calculation #total
        prob = np.exp(scoresF) / np.sum(np.exp(scoresF), keepdims=True, axis=1) # probability scores calculation

        # back propagation

        tmp = prob.copy() #using the prev

        tmp[range(len(y)), y] -= 1 # calculating through derivate of softmax formula
        tmp /= len(y) # normalizing

        da2=tmp.copy()
        dz2=da2.copy()

        dz2[[da2<0]]= .01
        dz2[[da2>=0]]=1
        dz2*=da2

        grads['w2'] = np.dot(a1.T, dz2)
        grads['w2'] += (2*reg * w2) # computing the gradient for  w1
        grads['b2'] = np.sum(dz2, axis=0) # computing the gradient for b2
        ###?
        da1 = np.dot(dz2, w2.T) # dot product of hidden layer and w2
        #dz1 = np.maximum(da1,0.01*da1) # leaky relu at output
        dz1=da1.copy()
        dz1[[da1<0]]=.01
        dz1[[da1>=0]]=1
        dz1*=da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['w1'] += (2*reg * w1) # computing the gradient for w1
        grads['b1'] = np.sum(dz1, axis=0) # computing the gradient for b1

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        num_train = x.shape[0]
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None

            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            sample = np.arange(num_train)
            np.random.shuffle(sample)
            sample = sample[0:batchSize - 1]
            Xbatch = x[sample, :]
            Ybatch = y[sample]

            # Compute loss and gradients using the current minibatch

            loss, grads = self.calLoss(Xbatch, y=Ybatch, reg=reg)
            lossHistory.append(loss)

            # Updating weights and biases with grdaients for each layer
            
            self.params['w1'] += -lr * grads['w1']
            self.params['w2'] += -lr * grads['w2']
            self.params['b1'] += -lr * grads['b1']
            self.params['b2'] += -lr * grads['b2']

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        yPred = None
        ###########################################################################
        # Implement this function; it should be VERY simple!                      #
        ###########################################################################

        w1= self.params['w1']
        b1=self.params['b1']
        w2= self.params['w2']
        b2=self.params['b2']

        hidden = np.dot(x, w1) + b1  # hidden layer calculation
        hidden=np.maximum(hidden,0.01*hidden) # leaky relu
        scores = np.dot(hidden, w2) + b2 # final scores calculation
        scores=np.maximum(scores,0.01*scores) # leaky relu
        yPred = np.argmax(scores, axis=1) # prediction
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        # compares the classes assigned in the predicted vs the correct tag
        acc = (np.mean(self.predict(x) == y)) * 100  # acc= average no. of correct hits *100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



