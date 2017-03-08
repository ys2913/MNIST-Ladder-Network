import torch
import numpy as np
import torch.nn as nn
import constants as c
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Affine(nn.Module):
    def __init__(self, length):
        super(Affine, self).__init__()
        self.length = length
        self.gamma = nn.Parameter(torch.rand(1, self.length))
        self.beta = nn.Parameter(torch.rand(1, self.length))

    def forward(self, inp):
        out = self.gamma.repeat(inp.size(0),1)*inp + self.beta.repeat(inp.size(0),1)
        return out

class Combinator(nn.Module):
    def __init__(self, length):
        super(Combinator, self).__init__()
        self.length = length
        self.a1 = nn.Parameter(torch.rand(1, self.length))
        self.a2 = nn.Parameter(torch.rand(1, self.length))
        self.a3 = nn.Parameter(torch.rand(1, self.length))
        self.a4 = nn.Parameter(torch.rand(1, self.length))
        self.a5 = nn.Parameter(torch.rand(1, self.length))
        self.a6 = nn.Parameter(torch.rand(1, self.length))
        self.a7 = nn.Parameter(torch.rand(1, self.length))
        self.a8 = nn.Parameter(torch.rand(1, self.length))
        self.a9 = nn.Parameter(torch.rand(1, self.length))
        self.a10 = nn.Parameter(torch.rand(1, self.length))

    def forward(self, zl, ul):
        u = self.a1.repeat(zl.size(0),1) * F.sigmoid(self.a2.repeat(zl.size(0),1)*ul + self.a3.repeat(zl.size(0),1)) \
            + self.a4.repeat(zl.size(0),1)*ul + self.a5.repeat(zl.size(0),1)
        v = self.a6.repeat(zl.size(0),1) * F.sigmoid(self.a7.repeat(zl.size(0),1)*ul + self.a8.repeat(zl.size(0),1)) \
            + self.a9.repeat(zl.size(0),1)*ul + self.a10.repeat(zl.size(0),1)
        out = (zl - u)*v + u
        return out

class Combinator2(nn.Module):
    def __init__(self, length):
        super(Combinator2, self).__init__()
        self.length = length
        self.bias_0 = nn.Parameter(torch.randn(1, self.length))
        self.bias_1 = nn.Parameter(torch.randn(1, self.length))
        self.w_0z = nn.Parameter(torch.randn(1, self.length))
        self.w_1z = nn.Parameter(torch.randn(1, self.length))
        self.w_0u = nn.Parameter(torch.randn(1, self.length))
        self.w_1u = nn.Parameter(torch.randn(1, self.length))
        self.w_0zu = nn.Parameter(torch.randn(1, self.length))
        self.w_1zu = nn.Parameter(torch.randn(1, self.length))
        self.w_sig = nn.Parameter(torch.randn(1, self.length))

    def forward(self, zl, ul):

        #print(str(zl.size())+" "+str(ul.size())+" "+str(self.bias_1.repeat(zl.size(0), 1).size()))
        tem = self.bias_1.repeat(zl.size(0), 1) + self.w_1z.repeat(zl.size(0), 1) * zl \
              + self.w_1u.repeat(zl.size(0), 1) * ul + self.w_1zu.repeat(zl.size(0), 1) * zl * ul
        out = self.bias_0.repeat(zl.size(0), 1) + self.w_0z.repeat(zl.size(0), 1) * zl + \
              self.w_0u.repeat(zl.size(0), 1) * ul + self.w_0zu.repeat(zl.size(0), 1) * zl * ul + tem.sigmoid_()
        return out

class Ladder(nn.Module):
    def __init__(self):
        super(Ladder, self).__init__()
        self.test = False
        self.noise = True
        # storing noise_encoder outputs
        self.e_noise_out = 0
        self.layer_out = []
        # storing clean_encoder outputs
        self.batch_mean = []
        self.batch_std = []
        self.batch_norm = []
        # storing decoder outputs
        self.decoder_out = [[], [], [], [], [], [], []]

        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(250, 10)

        self.tfc1 = nn.Linear(1000, 784)
        self.tfc2 = nn.Linear(500, 1000)
        self.tfc3 = nn.Linear(250, 500)
        self.tfc4 = nn.Linear(250, 250)
        self.tfc5 = nn.Linear(250, 250)
        self.tfc6 = nn.Linear(10, 250)

        self.af1 = Affine(1000) #nn.BatchNorm1d(1000)
        self.af2 = Affine(500)
        self.af3 = Affine(250)
        self.af4 = Affine(250)
        self.af5 = Affine(250)
        self.af6 = Affine(10)

        self.abn0 = nn.BatchNorm1d(784, affine=False)
        self.abn1 = nn.BatchNorm1d(1000, affine=False)
        self.abn2 = nn.BatchNorm1d(500, affine=False)
        self.abn3 = nn.BatchNorm1d(250, affine=False)
        self.abn4 = nn.BatchNorm1d(250, affine=False)
        self.abn5 = nn.BatchNorm1d(250, affine=False)
        self.abn6 = nn.BatchNorm1d(10, affine=False)

        self.comb0 = Combinator(784)
        self.comb1 = Combinator(1000)
        self.comb2 = Combinator(500)
        self.comb3 = Combinator(250)
        self.comb4 = Combinator(250)
        self.comb5 = Combinator(250)
        self.comb6 = Combinator(10)

        # self.relu = nn.ReLU()F.relu(x)
    def initialize(self):
        self.e_noise_out = 0
        self.layer_out = []
        # storing clean_encoder outputs
        self.batch_mean = []
        self.batch_std = []
        self.batch_norm = []
        # storing decoder outputs
        self.decoder_out = [[], [], [], [], [], [], []]

    def addgaussianNoise(self, x):
        noise = torch.Tensor(np.random.normal(c.MEAN, c.STD, x.size()))
        if torch.cuda.is_available():
            noise = noise.cuda() 
        x.data = x.data + noise 
        return x

    def noise_encoder(self, x):
        x = x.view(-1, 784)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())

        x = self.fc1(x)
        x = self.abn1(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af1.forward(x)  # Custom function needed
        x = F.relu(x)

        x = self.fc2(x)
        x = self.abn2(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af2.forward(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.abn3(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af3.forward(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.abn4(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af4.forward(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = self.abn5(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af5.forward(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = self.abn6(x)
        x = self.addgaussianNoise(x)
        self.layer_out.append(x.clone())
        x = self.af6.forward(x)
        x = F.relu(x)

        return x

    def clean_encoder(self, x):
        x = x.view(-1, 784)
        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn0(y))
        self.batch_norm.append(y)

        x = self.fc1(x)
        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn1(y))
        self.batch_norm.append(y)
        x = self.abn1(x)
        x = self.af1.forward(x)  # Note update x=abn1(x), before replacing self.bn
        x = F.relu(x)

        x = self.fc2(x)

        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn2(y))
        self.batch_norm.append(y)
        x = self.abn2(x)
        x = self.af2.forward(x)
        x = F.relu(x)

        x = self.fc3(x)

        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn3(y))
        self.batch_norm.append(y)
        x = self.abn3(x)
        x = self.af3.forward(x)
        x = F.relu(x)

        x = self.fc4(x)
        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn4(y))
        self.batch_norm.append(y)
        x = self.abn4(x)
        x = self.af4.forward(x)
        x = F.relu(x)

        x = self.fc5(x)
        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn5(y))
        self.batch_norm.append(y)
        x = self.abn5(x)
        x = self.af5.forward(x)
        x = F.relu(x)

        x = self.fc6(x)

        self.batch_mean.append(x.mean(dim=0))
        self.batch_std.append(x.var(dim=0))
        y = x.clone().detach()
        y = F.sigmoid(self.abn6(y))
        self.batch_norm.append(y)
        x = self.abn6(x)
        x = self.af6.forward(x)
        x = F.relu(x)

        return x

    def bnormalize(self, inp, mean, std):

        std = torch.sqrt(std + 1e-8)
        inp = (inp - mean.repeat(inp.size(0), 1)) / (std.repeat(inp.size(0), 1) )

        # mask = std == 0
        # #mask = mask.float()
        # inp[mask.repeat(inp.size(0),1)] = 0 

        return inp

    def decoder(self, x):
        # DECODER
        x = self.abn6(x)
        x = self.comb6.forward(self.layer_out[6], x)
        self.decoder_out[6] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[6], self.batch_std[6]))

        x = self.tfc6(x)
        x = F.relu(x)
        x = self.abn5(x)
        x = self.comb5.forward(self.layer_out[5], x)
        self.decoder_out[5] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[5], self.batch_std[5]))

        x = self.tfc5(x)
        x = F.relu(x)
        x = self.abn4(x)
        x = self.comb4.forward(self.layer_out[4], x)
        self.decoder_out[4] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[4], self.batch_std[4]))

        x = self.tfc4(x)
        x = F.relu(x)
        x = self.abn3(x)
        x = self.comb3.forward(self.layer_out[3], x)
        self.decoder_out[3] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[3], self.batch_std[3]))

        x = self.tfc3(x)
        x = F.relu(x)
        x = self.abn2(x)
        x = self.comb2.forward(self.layer_out[2], x)
        self.decoder_out[2] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[2], self.batch_std[2]))

        x = self.tfc2(x)
        x = F.relu(x)
        x = self.abn1(x)
        x = self.comb1.forward(self.layer_out[1], x)
        self.decoder_out[1] = F.sigmoid(self.bnormalize(x.clone(), self.batch_mean[1], self.batch_std[1]))

        x = self.tfc1(x)
        x = F.relu(x)
        x = self.comb0.forward(self.layer_out[0], x)
        self.decoder_out[0] = F.sigmoid(self.bnormalize(x, self.batch_mean[0], self.batch_std[0]))

        return x

    def forward(self, x):
        #print("Input: "+str(x.size()))
        self.initialize()
        #   ENCODER
        clean_x = x.clone()

        x = self.noise_encoder(x)
        self.e_noise_out = F.log_softmax(x.clone())

        clean_x = self.clean_encoder(clean_x)

        if self.test:
            return F.log_softmax(clean_x)

        x = self.decoder(x)
        return x#F.log_softmax(x)

    def setReqNoise(self, value):
        self.noise = value

    def getValues(self):
        output = []
        output.append(self.e_noise_out)
        output.append(self.batch_norm)
        output.append(self.decoder_out)
        return output

    def setTest(self, value):
        self.test = value
