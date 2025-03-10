import torch
import torch.nn as nn
import torch.optim as optim


class LSPLM(nn.Module):
    def __init__(self, m, optimizer, penalty='l2', batch_size=32, epoch=100, learning_rate=0.1, verbose=False):
        super(LSPLM, self).__init__()
        self.m = m
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.penalty = penalty

        self.softmax = None
        self.logistic = None

        self.loss_fn = nn.BCELoss(reduction='mean')

    def fit(self, X, y):
        if self.softmax is None and self.logistic is None:
            self.softmax = nn.Sequential(
                nn.Linear(X.shape[1], self.m).double(),
                nn.Softmax(dim=1).double()
            )

            self.logistic = nn.Sequential(
                nn.Linear(X.shape[1], self.m, bias=True).double()
                , nn.Sigmoid())

            if self.optimizer == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            elif self.optimizer == 'SGD':
                self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.1,
                                           nesterov=True)

        # noinspection DuplicatedCode
        for epoch in range(self.epoch):

            start = 0
            end = start + self.batch_size
            while start < X.shape[0]:

                if end >= X.shape[0]:
                    end = X.shape[0]

                X_batch = torch.from_numpy(X[start:end, :])
                y_batch = torch.from_numpy(y[start:end]).reshape(1, end - start)

                y_batch_pred = self.forward(X_batch).reshape(1, end - start)
                loss = self.loss_fn(y_batch_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                start = end
                end += self.batch_size

            if self.verbose and epoch % (self.epoch / 20) == 0:
                print('EPOCH: %d, loss: %f' % (epoch, loss))
        return self

    def forward(self, X):
        logistic_out = self.logistic(X)
        softmax_out = self.softmax(X)
        combine_out = logistic_out.mul(softmax_out)
        return combine_out.sum(1)

    def predict_proba(self, X):
        X = torch.from_numpy(X)
        return self.forward(X)

    def predict(self, X):
        X = torch.from_numpy(X)
        out = self.forward(X)
        out[out >= 0.5] = 1.0
        out[out < 0.5] = 0.0
        return out
