import numpy as np


class NearestNeighbors:

    def __init__(self):


    def train(self,x,y):
        self.xtr = x
        self.ytr = y

    def predict(self,x):

        num_test = x.shape[0]
        ypred = np.zeros(num_test, dtype = self. ytr.dtype)

        for i in xrange(num_test):

            distances = np.sum(np.abs(self.xtr - x[i.:]),axis=1)

            min_index = np.argmin(distances)
            ypred[i] = self.ytr[min_index]


        return ypred