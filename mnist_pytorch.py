from __future__ import print_function
import pickle
import constants as c
from ladder import Ladder
from dataloader import Loader
from model import AEMnist
import argparse
import torch
import result as re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', action='store', default='model.p',
                    help='modelname')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

data_loader = Loader(c.FILE_TRAIN_LABELED, c.FILE_TRAIN_UNLABELED, c.FILE_VALIDATION, c.FILE_TEST, kwargs)
train_loader = data_loader.getLabeledtrain()
unlabeled_train_loader = data_loader.getUnlabeledtrain()
valid_loader = data_loader.getValidation()

model = AEMnist()

if args.cuda:
    model.cuda()

l2loss = torch.nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train_unlabeled(epoch):
    model.train()
    model.setsupervised(False)
    for batch_idx, (data, target) in enumerate(unlabeled_train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)

        loss = l2loss(output,data)

        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Unsupervised Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_train_loader.dataset),
                       100. * batch_idx / len(unlabeled_train_loader), loss.data[0]))


def train(epoch):
    model.train()
    model.setsupervised(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch, valid_loader):
    model.eval()
    model.setsupervised(True)
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

#args.epochs = 1
for epoch in range(1, args.epochs + 1):
    train_unlabeled(epoch)
    test(epoch, valid_loader)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch, valid_loader)

modelname = c.MODEL_DIR + args.model + ".p"
re.makecsv(args.model, model, False)
torch.save(model.state_dict(), modelname)
