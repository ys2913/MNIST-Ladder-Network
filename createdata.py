from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import constants as c
import torch.nn.functional as F
import torch.optim as optim
from dataaug import DataAug
from torchvision import datasets, transforms
from torch.autograd import Variable
from sub import subMNIST       # testing the subclass of MNIST dataset

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])

trainset_original = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)

train_label_index = []
valid_label_index = []
for i in range(10):
    train_label_list = trainset_original.train_labels.numpy()
    label_index = np.where(train_label_list == i)[0]
    label_subindex = list(label_index[:300])
    valid_subindex = list(label_index[300: 1000 + 300])
    train_label_index += label_subindex
    valid_label_index += valid_subindex

trainset_np = trainset_original.train_data.numpy()
trainset_label_np = trainset_original.train_labels.numpy()

train_data_sub = trainset_np[train_label_index]
train_labels_sub = trainset_label_np[train_label_index]


da = DataAug()

augmented_data, augmented_label = da.dataaug(train_data_sub[0],train_labels_sub[0])
for i in range(1,train_data_sub.shape[0]):
    tdata, tlabel = da.dataaug(train_data_sub[i],train_labels_sub[i])
    augmented_data = np.append(augmented_data,tdata,axis=0)
    augmented_label = np.append(augmented_label,tlabel)

train_data_sub = np.append(train_data_sub,augmented_data,axis=0)
train_labels_sub = np.append(train_labels_sub, augmented_label,axis=0)

augdata = train_data_sub
auglabel = train_labels_sub
print(augdata.shape)
print(auglabel.shape)

train_data_sub = torch.from_numpy(augdata)
train_labels_sub = torch.from_numpy(auglabel)
print(train_labels_sub.size())
print(train_data_sub.size())

trainset_new = subMNIST(root='./data', train=True, download=True, transform=transform, k=18000)
trainset_new.train_data = train_data_sub.clone()
trainset_new.train_labels = train_labels_sub.clone()

pickle.dump(trainset_new, open("data/train_labeled_aug.p", "wb"))