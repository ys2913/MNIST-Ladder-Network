from __future__ import print_function
import pickle
import constants as c
from pladder import Ladder
from dataloader import Loader
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='Ladder')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs-supervised', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs-unsupervised', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

data_loader = Loader('data/train_labeled_aug.p', c.FILE_TRAIN_UNLABELED, c.FILE_VALIDATION, c.FILE_TEST, kwargs)
train_loader = data_loader.getLabeledtrain()
unlabeled_train_loader = data_loader.getUnlabeledtrain()
valid_loader = data_loader.getValidation()

model = Ladder()

if args.cuda:
    model.cuda()

l2loss = torch.nn.BCELoss() #torch.nn.L1Loss() # BCELoss : Pass through sigmoid 
#l2_2 = torch.nn.L1Loss()
nllloss = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train_unlabeled(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)

        values = model.getValues()
        decoder_out = values[2]
        clean_encoder_out = values[1]

        loss = c.lam[0] * l2loss(decoder_out[0], clean_encoder_out[0]) + \
               c.lam[1] * l2loss(decoder_out[1], clean_encoder_out[1]) + c.lam[2] * l2loss(decoder_out[2],
                                                                                           clean_encoder_out[2]) + \
               c.lam[3] * l2loss(decoder_out[3], clean_encoder_out[3]) + c.lam[4] * l2loss(decoder_out[4],
                                                                                           clean_encoder_out[4]) + \
               c.lam[5] * l2loss(decoder_out[5], clean_encoder_out[5]) + c.lam[6] * l2loss(decoder_out[6],
                                                                                           clean_encoder_out[6])
        
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Unsupervised Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_train_loader.dataset),
                       100. * batch_idx / len(unlabeled_train_loader), loss.data[0]))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        values = model.getValues()
        n_enc_out = values[0]
        decoder_out = values[2]
        clean_encoder_out = values[1]
        loss = 1000*F.nll_loss(n_enc_out, target) + c.lam[0] * l2loss(decoder_out[0], clean_encoder_out[0]) + \
               c.lam[1] * l2loss(decoder_out[1], clean_encoder_out[1]) + c.lam[2] * l2loss(decoder_out[2],
                                                                                           clean_encoder_out[2]) + \
               c.lam[3] * l2loss(decoder_out[3], clean_encoder_out[3]) + c.lam[4] * l2loss(decoder_out[4],
                                                                                           clean_encoder_out[4]) + \
               c.lam[5] * l2loss(decoder_out[5], clean_encoder_out[5]) + c.lam[6] * l2loss(decoder_out[6],
                                                                                           clean_encoder_out[6])
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch, valid_loader):
    model.eval()
    model.setTest(True)
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # encoder_out, en, de = model.getValues()
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size
    model.setTest(False)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))



for epoch in range(1, args.epochs_unsupervised+1):
    train_unlabeled(epoch)
    test(epoch, valid_loader)
    

for epoch in range(1, args.epochs_supervised+1):
    train(epoch)
    test(epoch, valid_loader)


torch.save(model, "ladder-60.p")
   
