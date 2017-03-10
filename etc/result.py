import torch
import pickle
import numpy as np
import pandas as pd
import constants as c
from dataloader import Loader
from torch.autograd import Variable


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
    mnist_model.setsupervised(True)
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

