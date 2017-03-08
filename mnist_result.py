from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import Loader
import pandas as pd
from torch.autograd import Variable


filename = 'model_params.p'
csvfilename = 'predictions.csv'

cuda = False
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 28 * 28 goes to 24 * 24 followed by 12 * 12 on maxpooling
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 12 * 12 goes to 8 * 8 followed by 4 * 4 on maxpooling
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)   # 4 * 4 * 20 = 320
        self.fc2 = nn.Linear(50, 10)

        # Decoder
        self.dfc2 = nn.Linear(10, 50)
        self.dfc1 = nn.Linear(50, 320)
        self.dconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.dconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.supervised = False

    def forward(self, x):
        x, indices1 = F.max_pool2d(self.conv1(x), 2, return_indices=True)
        x = F.relu(x)
        x, indices2 = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, return_indices=True)
        # x, indices2 = F.max_pool2d(self.conv2(x), 2, return_indices=True)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.supervised:
            return F.log_softmax(x)

        # Decoder
        x = self.dfc2(x)
        x = self.dfc1(x)
        x = x.view(-1, 20, 4, 4)
        x = F.relu(self.dconv2(F.max_unpool2d(x, kernel_size=2, indices=indices2, stride=2)))
        x = F.relu(self.dconv1(F.max_unpool2d(x, kernel_size=2, indices=indices1, stride=2)))
        return x

    def set_supervised_flag(self,supervised):
        self.supervised = supervised

mnist_model = Net()
print("Loading Model: "+ filename)
params = torch.load(filename)
mnist_model.load_state_dict(params)
print("Model Loaded")

print("loading Test Data")
test_set = pickle.load(open('data/test.p', "rb"))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, **kwargs)
print("Loaded Test Data")

correct = 0
label_predict = np.array([])
mnist_model.set_supervised_flag(True)
for data, target in test_loader:
    mnist_model.eval()
    if cuda:
    	data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = mnist_model(data)
    temp = output.data.max(1)[1]
    pred = output.data.max(1)[1]
    correct += pred.eq(target.data).cpu().sum()
    label_predict = np.concatenate((label_predict, temp.cpu().numpy().reshape(-1)))

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))


predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)
filename = 'predictions_SaWaSa.csv'
predict_label.to_csv(filename, index=False)
print('Saved to file: ' + filename )
