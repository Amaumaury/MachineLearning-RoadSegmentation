import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np


class LogisticRegression:
    def __init__(self, degree, learning_rate, max_iter=np.inf, threshold=-np.inf):
        if max_iter == np.inf and threshold == -np.inf:
            raise
        self.learning_rate = learning_rate
        self.degree = degree
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X, y):
        x_enhanced = utils.polynomial_enhancement(X, self.degree)
        x_enhanced = Variable(torch.from_numpy(x_enhanced).type(torch.FloatTensor))
        y = np.reshape(y, (len(y), 1))
        y = Variable(torch.from_numpy(y).float())

        ncols = x_enhanced.size()[1]
        self.model = torch.nn.Sequential(nn.Linear(ncols, 1, bias=False), nn.Sigmoid())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.losses = []
        epoch, prev_loss, next_loss = 0, 0, np.inf
        converged = False
        loss_f = nn.MSELoss()

        while epoch < self.max_iter and not converged:
            prev_loss = next_loss

            regressed = self.model(x_enhanced)
            loss = loss_f(regressed, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            next_loss = loss.data[0]
            converged = np.abs(prev_loss - next_loss) < self.threshold

            if epoch % 100 == 0:
                print('iter', epoch, 'loss:', next_loss)

            if converged:
                print('Converged at iteration', epoch, 'with loss difference', abs(prev_loss - next_loss))
            epoch += 1
            self.losses.append(next_loss)

        self.loss = self.losses[-1]

    def predict(self, X):
        X = utils.polynomial_enhancement(X, self.degree)
        X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
        return self.model(X).round().data

