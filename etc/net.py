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
#import matplotlib.pyplot as plt
import constants as c

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs-unsupervised', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs-supervised', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-as', type=str, default="starter_ae", metavar='N',
                    help='Name of file in which to store model')
args = parser.parse_args()
args.cuda = False #not args.no_cuda and torch.cuda.is_available()

#print(args.save_as)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loss = []
train_accuracy = []
test_accuracy = []
validation_accuracy = []

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('loading data!')
#data_loader = Loader(c.FILE_TRAIN_LABELED_AUG, c.FILE_TRAIN_UNLABELED, c.FILE_VALIDATION, c.FILE_TEST, kwargs)
#train_loader = data_loader.getLabeledtrain()
#unlabeled_loader = data_loader.getUnlabeledtrain()
#valid_loader = data_loader.getValidation()

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

model = Net()
if args.cuda:
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train_unsupervised(epoch):
    model.set_supervised_flag(False)
    model.train()
    for batch_idx, (data,target) in enumerate(unlabeled_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        #data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_loader.dataset),
                100. * batch_idx / len(unlabeled_loader), loss.data[0]))

def train_supervised(epoch):
    model.set_supervised_flag(True)
    model.train()
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    train_accuracy.append((100.0 * correct) / len(train_loader.dataset))    
    train_loss.append(total_loss)

testing = False
def test(epoch, test_loader):
    model.set_supervised_flag(True)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if testing:
        test_accuracy.append((100.0 * correct) / len(test_loader.dataset))
    else:
        validation_accuracy.append((100.0 * correct) / len(test_loader.dataset))    
"""
for epoch in range(1, args.epochs_unsupervised + 1):
    train_unsupervised(epoch)

for epoch in range(1, args.epochs_supervised + 1):
    train_supervised(epoch)
    test(epoch, valid_loader)
    if epoch == 71:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

with open(args.save_as + "_train_acc.p", 'wb') as file:
    pickle.dump(train_accuracy, file)

with open(args.save_as + "_valid_acc.p", 'wb') as file:
    pickle.dump(validation_accuracy, file) 
"""
#with open(args.save_as + "_train_loss.p", 'wb') as file:
#    pickle.dump(train_loss, file)


def makecsv(file, model, loadfile):
    cuda = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    data_loader = Loader(c.FILE_TRAIN_LABELED_AUG, c.FILE_TRAIN_UNLABELED, c.FILE_TEST, 'data/test-labeled.p', kwargs)
    test_loader = data_loader.getTest()
    test_actual = data_loader.getValidation()
    label_predict = np.array([])

    mnist_model = model
    if loadfile:
        mnist_model = torch.load(model)
    correct = 0

    for data, target in test_loader:
        mnist_model.eval()
        data, target = Variable(data, volatile=True), Variable(target)
        output = mnist_model(data)
        temp = output.data.max(1)[1]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        label_predict = np.concatenate((label_predict, temp.numpy().reshape(-1)))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    filename = 'predictions/' + file + "-labeled.csv"
    predict_label.to_csv(filename, index=False)

    label_predict = np.array([])
    correct = 0

    for data, target in test_actual:
        mnist_model.eval()
        data, target = Variable(data, volatile=True), Variable(target)
        output = mnist_model(data)
        temp = output.data.max(1)[1]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        label_predict = np.concatenate((label_predict, temp.numpy().reshape(-1)))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    filename = 'predictions/' + file + "-unlabeled.csv"
    predict_label.to_csv(filename, index=False)

#torch.save(model, args.save_as + ".p")
#makecsv(args.save_as, model, False)


