import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim, update=0):
        self.params = dict()
        self.update = update
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)

    def calLoss (self, x, y, reg):
        grads = dict()

        # Forward pass to calculate loss
        tmp = x.dot(self.params['w1']) + self.params['b1']
        hOutput = np.maximum(0.01 * tmp, tmp)
        scores = hOutput.dot(self.params['w2']) + self.params['b2']
        scores = np.maximum(0.01 * scores, scores)
        scores -= np.max(scores, axis=1, keepdims=True)
        scores = np.exp(scores)
        scoresProbs = scores/np.sum(scores, axis=1, keepdims=True)
        logProbs = -np.log(scoresProbs[np.arange(x.shape[0]), y])
        loss = np.sum(logProbs) / x.shape[0]
        loss += 0.5 * reg * np.sum(self.params['w1'] * self.params['w1']) + 0.5 * reg * np.sum(self.params['w2'] * self.params['w2'])

        # Backward pass to calculate each gradient
        dScoresProbs = scoresProbs
        dScoresProbs[range(x.shape[0]), list(y)] -= 1
        dScoresProbs /= x.shape[0]
        grads['w2'] = hOutput.T.dot(dScoresProbs) + reg * self.params['w2']
        grads['b2'] = np.sum(dScoresProbs, axis=0)

        dhOutput = dScoresProbs.dot(self.params['w2'].T)
        dhOutputAct = (hOutput >= 0) * dhOutput + (hOutput < 0) * dhOutput * 0.01
        grads['w1'] = x.T.dot(dhOutputAct) + reg * self.params['w1']
        grads['b1'] = np.sum(dhOutputAct, axis=0)

        return loss, grads

    def train (self, x, y, lr=1e-3, reg=1e-5, iterations=100, batchSize=200, decay=0.95, verbose=False):
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
        lossHistory = []

        # Initialize value for each update optimizer
        self.params['VW2'] = 0
        self.params['VW1'] = 0
        self.params['cacheW2'] = 0
        self.params['cacheW1'] = 0

        for i in range(iterations):
            batchID = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[batchID]
            yBatch = y[batchID]
            loss, grads = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            if self.update == 0:
                #########################################################################
                # TODO: 10 points                                                       #
                # - Use Naive Update to update weight parameter                         #
                #########################################################################
                #formula:
                #W += -learning_rate * dW
                self.params['w1'] += -lr * grads['w1']
                self.params['w2'] += -lr * grads['w2']


            elif self.update == 1:
                #########################################################################
                # TODO: 10 points                                                       #
                # - Use Momentum Update to update weight parameter                      #
                # - Momentum = 0.9                                                      #
                #########################################################################
                #formula:
                # V = mu * V - learning_rate * dW
                # W += V
                self.params['VW1'] = 0.9 * self.params['VW1'] - lr * grads['w1']# V = mu * V - learning_rate * dW
                self.params['w1'] += self.params['VW1'] # W += V
                #similarly for w2
                self.params['VW2'] = 0.9 * self.params['VW2'] - lr * grads['w2']
                self.params['w2'] += self.params['VW2']


            elif self.update == 2:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use Nesterov Update to update weight parameter                      #
                # - Momentum = 0.9                                                      #
                # - Hint                                                                #
                #   v_prev = v                                                          #
                #   v = mu * v - lr * dw                                                #
                #   w += -mu * v_prev + (1 + mu) * v                                    #
                #########################################################################
                #formula:
                #   v_prev = v
                #   v = mu * v - lr * dw
                #   w += -mu * v_prev + (1 + mu) * v

                mu=0.9 #given

                self.params['cacheW1'] = self.params['VW1'] # v_prev = v
                self.params['VW1'] = mu * self.params['VW1'] - lr * grads['w1']# v = mu * v - lr * dw
                self.params['w1'] += -mu * self.params['cacheW1'] +((1.0+mu) * self.params['VW1'])  # w += -mu * v_prev + (1 + mu) * v

                #similarly for w2
                self.params['cacheW2'] = self.params['VW2']
                self.params['VW2'] = mu * self.params['VW2'] - lr * grads['w2']
                self.params['w2'] += -mu * self.params['cacheW2'] + ((1.0+mu) * self.params['VW2'])

            elif self.update == 3:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use AdaGrad Update to update weight parameter                       #
                #########################################################################
                #formula:
                # cache += dW ^ 2
                # W += -learning_rate * dW / (np.sqrt(cache) + 1e-7)

                eps=1e-7
                self.params['cacheW1'] += grads['w1']**2 # cache += dW ^ 2
                self.params['w1'] += -lr * grads['w1']/(np.sqrt(self.params['cacheW1'])+eps) # W += -learning_rate * dW / (np.sqrt(cache) + 1e-7)
                
                #similarly for w2
                self.params['cacheW2'] +=grads['w2']**2
                self.params['w2'] += -lr*grads['w2']/(np.sqrt(self.params['cacheW2'])+eps)

            elif self.update == 4:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use RMSProp Update to update weight parameter                       #
                #########################################################################
                #formula:
                #cache = decay * cache + (1-decay) * dW^2
                #W += -learning_rate*dW/(np.sqrt(cache)+1e-7)

                eps=1e-7
                self.params['cacheW1'] = decay * self.params['cacheW1'] + (1 - decay) * (grads['w1'] ** 2) #cache = decay * cache + (1-decay)*dW^2.
                self.params['w1'] += -lr * grads['w1'] / (np.sqrt(self.params['cacheW1']) + eps) # W += -learning_rate*dW/(np.sqrt(cache)+1e-7)
                
                #similarly for w2
                self.params['cacheW2'] = decay * self.params['cacheW2'] + (1 - decay) * (grads['w2'] ** 2)
                self.params['w2'] += -lr * grads['w2'] / (np.sqrt(self.params['cacheW2']) + eps)

            else:
                #########################################################################
                # TODO: 20 points                                                       #
                # - Use Adam Update to update weight parameter                          #
                # - B1 = 0.9, B2 = 0.999                                                #
                #########################################################################

                #formula:
                # v = B1 * v + (1 - B1) *dW
                # cache = B2 * cache + (1 - B2)*(dW ^ 2)  # gradient square
                # vb = v / (1 - B1 ^ t)  # Bias corrected gradient
                # cacheb = cache / (1 - B2 ^ t)  # Bias corrected gradient square
                # W -= learning_rate * vb / [np.sqrt(cacheb) + 1e-7]
                B1 = 0.9
                B2 = 0.999
                eps = 1e-7

                self.params['VW1']=B1*self.params['VW1']+(1.0-B1)*grads['w1'] # v = B1 * v + (1 - B1) *dW
                self.params['cacheW1']=B2*self.params['cacheW1']+(1.0-B2)*(grads['w1']**2) # cache = B2 * cache + (1 - B2)*(dW ^ 2)  # gradient square
                self.params['vb1']=self.params['VW1']/(1.0-(B1**(i+1))) # vb = v / (1 - B1 ^ t)  # Bias corrected gradient
                self.params['cache1b']=self.params['cacheW1']/(1-(B2**(i+1))) # cacheb = cache / (1 - B2 ^ t)  # Bias corrected gradient square
                self.params['w1'] -= lr * self.params['vb1'] / (np.sqrt(self.params['cache1b']) + eps) # W -= learning_rate * vb / [np.sqrt(cacheb) + 1e-7]

                #similarly for w2
                self.params['VW2']=B1*self.params['VW2']+(1-B1)*grads['w2']
                self.params['cacheW2']=B2*self.params['cacheW2']+(1-B2)*(grads['w2']**2)
                self.params['vb2']=self.params['VW2']/(1-(B1**(i+1)))
                self.params['cache2b']=self.params['cacheW2']/(1-(B2**(i+1)))
                self.params['w2'] -= lr * self.params['vb2'] / (np.sqrt(self.params['cache2b']) + eps)


            self.params['b2'] += -lr * grads['b2']
            self.params['b1'] += -lr * grads['b1']
            lr *= decay
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        tmp = x.dot(self.params['w1']) + self.params['b1']
        hOutput = np.maximum(0.01 * tmp, tmp)
        scores = hOutput.dot(self.params['w2']) + self.params['b2']
        yPred = np.argmax(scores, axis=1)
        return yPred

    def calAccuracy (self, x, y):
        acc = 100.0 * (np.sum(self.predict(x) == y) / float(x.shape[0]))
        return acc



