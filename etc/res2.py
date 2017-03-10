import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import constants as c
from dataloader import Loader
from torch.autograd import Variable


cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

data_loader = Loader(c.FILE_TRAIN_LABELED_AUG, c.FILE_TRAIN_UNLABELED, c.FILE_TEST, 'data/test-labeled.p', kwargs)
test_loader = data_loader.getTest()
test_actual = data_loader.getValidation()
label_predict = np.array([])



def callval(mnist_model, test_loader, test_actual, model, file):
    label_predict = np.array([])
    loadfile = True
    if loadfile:
        mnist = torch.load(model)

    mnist_model.load_state_dict(mnist)
    correct = 0
    if torch.cuda.is_available():
        mnist_model.cuda()
    mnist_model.setsupervised(True)
    for data, target in test_loader:
        mnist_model.eval()
        if cuda:
            data, target = data.cuda(),target.cuda()
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
    filename = 'predictions/' + file + "-labeled.csv"
    predict_label.to_csv(filename, index=False)

    label_predict = np.array([])
    correct = 0

    for data, target in test_actual:
        mnist_model.eval()
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = mnist_model(data)
        temp = output.data.max(1)[1]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        label_predict = np.concatenate((label_predict, temp.cpu().numpy().reshape(-1)))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    filename = 'predictions/' + file + "-unlabeled.csv"
    predict_label.to_csv(filename, index=False)
    return

class AEMnist(nn.Module):
    def __init__(self):
        super(AEMnist, self).__init__()
        self.supervised = False
        # ENCODER
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 50)
        self.indx1 = []
        self.indx2 = []
        self.fc3 = nn.Linear(50, 10)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.tfc1 = nn.Linear(200, 320)
        self.tfc2 = nn.Linear(50, 200)
        self.munpool2 = nn.MaxUnpool2d(2)
        self.munpool1 = nn.MaxUnpool2d(2)

        # Batchnorm
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(50)

    def initialize(self):
        self.indx1 = []
        self.indx2 = []

    def setsupervised(self, value):
        self.supervised = value;
        #print("Supervised Value Set = " + str(value))

    def encoder(self, x):
        x = self.conv1(x)   # 10 x 24 x 24
        x = self.bn1(x)
        x, self.indx1 = F.max_pool2d(x, 2, return_indices=True)     # 10 x 12 x 12
        x = F.relu(x)

        x = self.conv2(x)  #   20 x 8 x 8
        x = self.bn2(x)
        x, self.indx2 = F.max_pool2d(x, 2, return_indices=True) #   20 x 4 x 4
        x = F.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)

        return x

    def decoder(self, x):
        x = self.tfc2(x)
        x = F.relu(x)

        x = self.tfc1(x)
        x = F.relu(x)

        x = x.view(-1, 20, 4, 4)
        x = self.munpool2(x, self.indx2)
        x = self.tconv2(x)
        x = F.relu(x)
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

mnist_model = AEMnist()

callval(mnist_model, test_loader, test_actual, 'model/mnist_ae_bn_90-gpu.p', 'mnist_ae_bn_90-gpu')
#makecsv('mnist-ae-bn-90-gpu', 'model/mnist_ae_bn_90-gpu.p', True)

class AEMnist(nn.Module):
    def __init__(self):
        super(AEMnist, self).__init__()
        self.supervised = False
        # ENCODER
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 50)
        self.indx1 = []
        self.indx2 = []
        self.fc3 = nn.Linear(50, 10)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.tfc1 = nn.Linear(200, 320)
        self.tfc2 = nn.Linear(50, 200)
        self.munpool2 = nn.MaxUnpool2d(2)
        self.munpool1 = nn.MaxUnpool2d(2)

        # Batchnorm
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(50)

    def initialize(self):
        self.indx1 = []
        self.indx2 = []

    def setsupervised(self, value):
        self.supervised = value;
        #print("Supervised Value Set = " + str(value))

    def encoder(self, x):
        x = self.conv1(x)   # 10 x 24 x 24
        x = self.bn1(x)
        x, self.indx1 = F.max_pool2d(x, 2, return_indices=True)     # 10 x 12 x 12
        x = F.relu(x)

        x = self.conv2(x)  #   20 x 8 x 8
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x, self.indx2 = F.max_pool2d(x, 2, return_indices=True) #   20 x 4 x 4
        x = F.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)

        return x

    def decoder(self, x):
        x = self.tfc2(x)
        x = F.relu(x)

        x = self.tfc1(x)
        x = F.relu(x)

        x = x.view(-1, 20, 4, 4)
        x = self.munpool2(x, self.indx2)
        x = self.tconv2(x)
        x = F.relu(x)
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



mnist_model = AEMnist()

callval(mnist_model, test_loader, test_actual, 'model/mnist_ae_bn_drop_90-gpu.p', 'mnist_ae_bn_drop_90-gpu')
#makecsv('mnist-ae-bn-90-gpu', 'model/mnist_ae_bn_drop_90-gpu.p', True)

class AEMnist(nn.Module):
    def __init__(self):
        super(AEMnist, self).__init__()
        self.supervised = False
        # ENCODER
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 50)
        self.indx1 = []
        self.indx2 = []
        self.fc3 = nn.Linear(50, 10)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.tfc1 = nn.Linear(200, 320)
        self.tfc2 = nn.Linear(50, 200)
        self.munpool2 = nn.MaxUnpool2d(2)
        self.munpool1 = nn.MaxUnpool2d(2)

        # Batchnorm
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(50)

    def initialize(self):
        self.indx1 = []
        self.indx2 = []

    def setsupervised(self, value):
        self.supervised = value;
        #print("Supervised Value Set = " + str(value))

    def encoder(self, x):
        x = self.conv1(x)   # 10 x 24 x 24
        x = self.bn1(x)
        x, self.indx1 = F.max_pool2d(x, 2, return_indices=True)     # 10 x 12 x 12
        x = F.leaky_relu(x,0.1)

        x = self.conv2(x)  #   20 x 8 x 8
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x, self.indx2 = F.max_pool2d(x, 2, return_indices=True) #   20 x 4 x 4
        x = F.leaky_relu(x, 0.1)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.1)

        return x

    def decoder(self, x):
        x = self.tfc2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.tfc1(x)
        x = F.leaky_relu(x, 0.1)

        x = x.view(-1, 20, 4, 4)
        x = self.munpool2(x, self.indx2)
        x = self.tconv2(x)
        x = F.leaky_relu(x, 0.1)
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

mnist_model = AEMnist()
callval(mnist_model, test_loader, test_actual, 'model/mnist_ae_bn_lrelu_90-gpu.p', 'mnist_ae_bn_lrelu_90-gpu')
#makecsv('mnist-ae-bn-90-gpu', 'model/mnist_ae_bn_lrelu_90-gpu.p', True)

class AEMnist(nn.Module):
    def __init__(self):
        super(AEMnist, self).__init__()
        self.supervised = False
        # ENCODER
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 200)
        self.fc2 = nn.Linear(200, 50)
        self.indx1 = []
        self.indx2 = []
        self.fc3 = nn.Linear(50, 10)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.tfc1 = nn.Linear(200, 320)
        self.tfc2 = nn.Linear(50, 200)
        self.munpool2 = nn.MaxUnpool2d(2)
        self.munpool1 = nn.MaxUnpool2d(2)

    def initialize(self):
        self.indx1 = []
        self.indx2 = []

    def setsupervised(self, value):
        self.supervised = value;
        #print("Supervised Value Set = " + str(value))

    def encoder(self, x):
        x = self.conv1(x)   # 10 x 24 x 24
        x, self.indx1 = F.max_pool2d(x, 2, return_indices=True)     # 10 x 12 x 12
        x = F.relu(x)

        x = self.conv2(x)  #   20 x 8 x 8
        x, self.indx2 = F.max_pool2d(x, 2, return_indices=True) #   20 x 4 x 4
        x = F.relu(x)

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return x

    def decoder(self, x):
        x = self.tfc2(x)
        x = F.relu(x)

        x = self.tfc1(x)
        x = F.relu(x)

        x = x.view(-1, 20, 4, 4)
        x = self.munpool2(x, self.indx2)
        x = self.tconv2(x)
        x = F.relu(x)
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

mnist_model = AEMnist()
callval(mnist_model, test_loader, test_actual, 'model/mnist_ae_nodrop_90-gpu.p', 'mnist_ae_nodrop_90-gpu')
#makecsv('mnist-ae-bn-90-gpu', 'model/mnist_ae_nodrop_90-gpu.p', True)
