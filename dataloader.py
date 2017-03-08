import pickle
import torch


class Loader():
    def __init__(self, train, utrain, valid, test, kwargs):
        self.train_labeled = train
        self.train_unlabeled = utrain
        self.validation = valid
        self.test = test
        self.kwargs = kwargs

    def getLabeledtrain(self):
        print('Loading Labeled Training Data!')
        trainset_labeled = pickle.load(open(self.train_labeled, "rb"))
        train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **self.kwargs)
        print('Loaded Labeled Training Data!')
        #print(train_loader.train_data.size())
        return train_loader

    def getUnlabeledtrain(self):
        print('Loading Unlabeled Training Data!')
        trainset_unlabeled = pickle.load(open(self.train_unlabeled, "rb"))
        labels = torch.Tensor(trainset_unlabeled.train_data.size()[0])
        labels.fill_(0)
        trainset_unlabeled.train_labels = labels
        train_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True, **self.kwargs)
        print('Loaded Unlabeled Training Data!')
        #print(train_loader.train_data.size())
        return train_loader

    def getValidation(self):
        print('Loading Validation Data!')
        valid_set = pickle.load(open(self.validation, "rb"))
        train_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False, **self.kwargs)
        print('Loaded Validation Data!')
        #print(train_loader.train_data.size())
        return train_loader

    def getTest(self):
        print('Loading Test Data!')
        test_set = pickle.load(open(self.test, "rb"))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, **self.kwargs)
        print('Loaded Test Data!')
        #print(test_loader.train_data.size())
        return test_loader


