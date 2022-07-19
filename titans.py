"""
AF
users can either just train a model on the titanic dataset
or they can train and evaluate their model by adding an addition argument:
the testing dataset. 

This program will have the option of being run with ray.io
"""
import ray
from ray import workflow
# import ray

from random import random
import numpy as np 
import math
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch.nn.functional as F

from titan_dataset import TitanData_Train
from titan_dataset import TitanData_Test

import sys

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.label = nn.Sequential(
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.label(x)

""" 
input: the training data_loader
output: the average loss for each batch and the total correct 
performs gradient descent based on input training data. with BCELoss
"""
@workflow.step
def train_fn(train_loader) -> float:

    model.train() 
    size = len(train_loader.dataset)
    for batch, (X, Y) in enumerate(train_loader):
        if len(X)!=batch_size: continue

        X = X.float()
        Y = Y.float()
        pred_Y = model(X)
        print(Y)
        print(pred_Y)

        loss = loss_fn(pred_Y, Y)

        #optimize grad descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch%10 == 0:
        loss, current = loss.item(), batch*len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

""" 
input: the validation data_loader
output: the average loss for each batch and the total correct 
"""
@workflow.step
def valid_fn(valid_loader):
    total_loss = 0
    model.eval()
    tot_correct = 0
    with torch.no_grad():
        for batch, (X_val, Y_val) in enumerate(valid_loader):
            if len(X_val)!=batch_size: continue
            X_val = X_val.float()
            Y_val = Y_val.float()
            pred_Y_val = model(X_val)

            pred_labels = pred_Y_val.argmax(1)
            gt_labels = Y_val.argmax(1)
            
            # print(gt_labels, pred_labels)
            assert(len(gt_labels) == len(pred_labels))
            for i in range(len(pred_labels)):
                if pred_labels[i] == gt_labels[i]: tot_correct += 1

            loss = loss_fn(pred_Y_val, Y_val)
            total_loss+=loss
    return total_loss / len(valid_loader), tot_correct

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print("USAGE: <training-data>\n\
            or <training-data> <testing-data>")
        exit(1)
    else:
        train_csv = sys.argv[1]
        test_csv = sys.argv[2]
        print(train_csv, test_csv)

    """ Import data """
    torch.autograd.set_detect_anomaly(True)
    data_train = TitanData_Train(train_csv)
    data_test = TitanData_Test(test_csv)

    #splitting training data
    train_size = int(len(data_train) * 0.8)
    valid_size = len(data_train) - train_size
    data_train, data_valid = data.random_split(data_train, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    #definitions
    epochs = 10
    batch_size = 32
    #hyper parameter for loss function 
    L = 1
    
    #define model and loss fn
    model = RNN()
    loss_fn = torch.nn.BCELoss()
    lr = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr)

    #change data format to dataloader
    train_loader = DataLoader(data_train, batch_size = batch_size)
    valid_loader = DataLoader(data_valid, batch_size = batch_size)
    test_loader = DataLoader(data_test, batch_size = 1)

    workflow.init()

    for ep in range(epochs):
        print(f"Epoch: {ep+1}\n------------------------")

        """
        The two functions below run in parallel
        """
        train_fn.step(train_loader).run()
        avg_batch_loss, tot_correct = valid_fn.step(valid_loader).run()
        
    print(f"Average Validation Loss for Epoch {ep+1}: {avg_batch_loss}")
    print(f"Accuracy of Epoch {ep+1}: {tot_correct} correct out of {batch_size*len(valid_loader)}")

        #Predicting

    # #split training data to train and validation
    # X, Y = data_train.getfnl()
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.33, random_state = 32)

    

