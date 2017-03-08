from __future__ import print_function
import pickle
import constants as c
from dataloader import Loader
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

class AEFnet(nn.Module):
    def __init__(self):
        super(AEFnet, self).__init__()
        self.supervised = False
        # ENCODER
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 50)
        self.indx1 = []
        self.indx2 = []
        self.fc3 = nn.Linear(50, 10)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(32, 1, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5)
        self.tfc1 = nn.Linear(120, 256)
        self.tfc2 = nn.Linear(50, 120)
        self.munpool2 = nn.MaxUnpool2d(2)
        self.munpool1 = nn.MaxUnpool2d(3)

        # Batchnorm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(50)

    def initialize(self):
        self.indx1 = []
        self.indx2 = []

    def setsupervised(self, value):
        self.supervised = value;
        #print("Supervised Value Set = " + str(value))

    def encoder(self, x):
        x = self.conv1(x)   # 32 x 24 x 24
        x = self.bn1(x)
        x, self.indx1 = F.max_pool2d(x, 3, return_indices=True)     # 32 x 8 x 8
        x = F.relu(x)

        x = self.conv2(x)  #   64 x 4 x 4
        x = self.bn2(x)
        x, self.indx2 = F.max_pool2d(self.conv2_drop(x), 2, return_indices=True) #   64 x 2 x 2
        x = F.relu(x)

        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)

        return x

    def decoder(self, x):
        x = self.tfc2(x)
        x = F.relu(x)

        x = self.tfc1(x)
        x = F.relu(x)

        x = x.view(-1, 64, 2, 2)
        x = self.munpool2(x, self.indx2) # 64 x 4 x 4
        #print(x.size())
        x = self.tconv2(x)  # 64 x 8 x 8
        x = F.relu(x)
        #print(x.size())
        x = self.munpool1(x, self.indx1)
        x = self.tconv1(x)
        return x

    def forward(self, x):
        x = self.encoder(x)

        if self.supervised:
            x = self.fc3(x)
            return F.log_softmax(x)

        x = self.decoder(x)
        return x



model = AEFnet()

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
