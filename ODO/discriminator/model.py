#!/usr/local/bin/env python

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import linregress
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden5 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden6 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        if not self.training:
            x = torch.from_numpy(x).float() 
            x = Variable(x.cuda())
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = self.predict(x)             # linear output
        return x

    def train_network(self, loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = torch.nn.L1Loss().cuda()  # this is for regression mean squared loss
        # reduces lr as learning levels off
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        for epoch in range(0, 20):
            # train the network
            total_loss = 0.0
            for step, (x, y) in tqdm(enumerate(loader), total=len(loader)):
                prediction = self(x)  # input x and predict based on x
                loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # back propagation, compute gradients
                optimizer.step()  # apply gradients
                total_loss += float(loss)
            scheduler.step(total_loss)
            print('Loss = {}'.format(total_loss))

    def save_ckpt(self):
        torch.save(self.state_dict(), './net.ckpt')

    def get_r_squared(self, testing_data):
        x = [np.array(x[0]) for x in testing_data]
        y = np.array([x[1][0] for x in testing_data])
        predict = [self(vec).cpu().detach().numpy() for vec in x]
        predict = np.array([x[0] for x in predict])
        slope, intercept, r_value, p_value, std_err = linregress(y, predict)
        print("R-squared: %f" % r_value ** 2)
    





