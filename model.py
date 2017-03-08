import torch
import torch.nn as nn
import torch.nn.functional as F
from ladder import Affine
from ladder import Combinator
import torch.optim as optim
from torch.autograd import Variable


class MnistBNP(nn.Module):
    def __init__(self):
        super(MnistBNP, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(50)

    def decoder(self, x):
        #   Initialization
        self.tconv1 = nn.ConvTranspose2d(1, 10, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(10, 20, kernel_size=5)
        # self.tconv2_drop = nn.Dropout2d()
        self.tfc1 = nn.Linear(200, 320)
        self.tfc2 = nn.Linear(50, 100)
        self.tfc3 = nn.Linear(10, 50)
        self.tbn1 = nn.BatchNorm2d(10)
        self.tbn2 = nn.BatchNorm2d(20)
        self.tbn3 = nn.BatchNorm1d(320)
        self.tbn4 = nn.BatchNorm1d(200)
        self.tbn5 = nn.BatchNorm1d(50)

        #
        x = self.tbn5(self.tfc3(x))
        x = F.relu(x)
        x = self.tbn4(self.tfc2(x))
        x = F.relu(x)
        x = self.tbn3(self.tfc1(x))
        x = F.relu(x)
        x = x.view(-1, )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)


class MnistBN(nn.Module):
    def __init__(self):
        super(MnistBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


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

        x = self.conv2_drop(self.conv2(x))  #   20 x 8 x 8
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


class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class SWWAE(nn.Module):
    def __init__(self):
        super(SWWAE, self).__init__()
        self.encoder = [[], [], []]
        self.decoder = [[], [], []]
        self.encoder_out = []
        self.test = False

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.tconv1 = nn.ConvTranspose2d(64, 1, 5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.tconv2 = nn.ConvTranspose2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.tconv3 = nn.ConvTranspose2d(64, 64, 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 10)
        self.fc2 = nn.Linear(10, 256)
        self.mpool = nn.MaxPool2d(2, return_indices=True)
        self.munpool = nn.MaxUnpool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, ind1 = self.mpool(self.relu(self.conv1(x)))
        self.encoder[0] = x.detach().clone()
        x, ind2 = self.mpool(self.relu(self.conv2(x)))
        self.encoder[1] = x.detach().clone()
        x, ind3 = self.mpool(self.relu(self.conv3(x)))
        self.encoder[1] = x.detach().clone()
        x = x.view(-1, 256)
        batch = x.size()[0]
        x = self.relu(self.fc1(x))

        if self.test:
            return F.log_softmax(x)
        else:
            self.encoder_out = F.log_softmax(x.clone())

        # decoder
        x = self.relu(self.fc2(x))
        x = x.view(batch, 64, 2, 2)
        self.decoder[2] = x.clone()
        x = self.relu(self.tconv3(self.munpool(x, ind3)))
        self.decoder[1] = x.clone()
        x = self.relu(self.tconv2(self.munpool(x, ind2)))
        self.decoder[0] = x.clone()
        x = self.tconv1(self.munpool(x, ind1))
        return x

    def getValues(self):
        return self.encoder_out, self.encoder, self.decoder

    def setTest(self, value):
        self.test = value


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder_out = []
        self.encoder = [[], []]
        self.decoder = [[], []]
        self.test = False

        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(2, return_indices=True)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 50)
        self.fc4 = nn.Linear(50, 320)
        self.munpool = nn.MaxUnpool2d(2)
        self.tconv2 = nn.ConvTranspose2d(20, 10, 5)
        self.tconv1 = nn.ConvTranspose2d(10, 1, 5)

    def forward(self, x):
        x, ind1 = self.mpool(self.relu(self.conv1(x)))
        self.encoder[0] = x.detach().clone()
        x, ind2 = self.mpool(self.conv2_drop(self.relu(self.conv2(x))))
        self.encoder[1] = x.detach().clone()
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.relu(self.fc2(x))

        if self.test:
            return F.log_softmax(x)
        else:
            self.encoder_out = F.log_softmax(x.clone())

        # decoder
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = x.view(-1, 20, 4, 4)
        self.decoder[1] = x.clone()
        x = self.relu(self.tconv2(self.munpool(x, ind2)))
        self.decoder[0] = x.clone()
        x = self.relu(self.tconv1(self.munpool(x, ind1)))
        return x

    def getValues(self):
        return self.encoder_out, self.encoder, self.decoder

    def setTest(self, value):
        self.test = value
