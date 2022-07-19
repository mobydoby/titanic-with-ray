from distutils.command.clean import clean
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
from scipy import stats

def clean_data(data):
    print(data[61][6])
    #changes sex identification from string to integer
    data[:,4] = np.where(data[:,4] == "male", 1, 2)

    #changes nan entries to random variable 
    m = stats.mode(data[:, 5])
    for i in range(len(data[:, 5])): 
        if np.isnan(data[i][5]): 
            data[i][5] = m[0][0]

    data[:,11] = np.where(data[:,11] == 'S', 1, data[:,11])
    data[:,11] = np.where(data[:,11] == 'Q', 2, data[:,11])
    data[:,11] = np.where(data[:,11] == 'C', 3, data[:,11])

    s = stats.mode(data[:, 11])
    for i in range(len(data[:, 11])): 
        if np.isnan(data[i][11]): 
            print(f"boo {data[i][11]}")
            data[i][11] = s[0][0]

    X = np.column_stack((data[:, 2],data[:, 4:8], data[:, 9], data[:,11])).astype(np.float)
    
    labels = data[:, 1].astype(np.float)
    Y = np.zeros((data.shape[0], 2))
    for i in range(len(labels)):
        Y[i][int(labels[i])] = 1
    
    return torch.tensor(X), torch.tensor(Y)

def clean_data_test(data):
    #changes sex identification from string to integer
    data[:,3] = np.where(data[:,3] == "male", 1, 2)

    #changes nan entries to random variable 
    m = stats.mode(data[:, 4])
    for i in range(len(data[:, 4])): 
        if np.isnan(data[i][4]): data[i][4] = m[0][0]

    data[:,10] = np.where(data[:,10] == 'S', 1, data[:,10])
    data[:,10] = np.where(data[:,10] == 'Q', 2, data[:,10])
    data[:,10] = np.where(data[:,10] == 'C', 3, data[:,10])

    X = np.column_stack((data[:, 1],data[:, 3:7], data[:, 8], data[:,10])).astype(np.float)

    return torch.tensor(X)

"""
Input: Takes in the csv file name of the testing set (this is a list of dictionaries)
Output: Dataset object with the following properties loaded from the csv file
    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    passenger_data: P x 11 tensor (P is the number of passengers)
    ground_truth: P x 1 tensor (did each passenger survive)
    
"""
class TitanData_Train(Dataset):
    def __init__(self, csv_file):
        """
        Constructor Args: csv file
        self.data 
        """
        self.data = pd.read_csv(csv_file).to_numpy()

        self.X, self.Y = clean_data(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        gets the value of each datavalue 
        """
        return self.X[idx], self.Y[idx]

    def getfnl(self):
        return self.X, self.Y

class TitanData_Test(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Constructor Args: 
        csv file
        transformation
        """
        self.info = pd.read_csv(csv_file).to_numpy()
        self.X = clean_data_test(self.info)
    
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        """
        gets the value of each datavalue 
        """
        return self.X[idx]

if __name__ == "__main__":
    trainset = TitanData_Train("./titanic/data/train.csv")
    print(len(trainset))
