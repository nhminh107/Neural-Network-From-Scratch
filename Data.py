import numpy as np
import torch
import torchvision.datasets as td
import random
import torchvision.transforms as transforms

class DataLoader: 
    def __init__(self, dataset, batch_size, shuffle = False): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self): 
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range (0, len(indices), self.batch_size): 
            #Lặp qua data với bước nhảy bằng batch_size 
            batch_indices = indices[i:i+self.batch_size]

            images = []
            labels = [] 

            for idx in batch_indices: 
                img, label = self.dataset[idx]
                images.append(img)
                labels.append(label)
            yield torch.stack(images), torch.tensor(labels)
class DataMNIST:     
    def __init__(self):
        self.data_train = td.MNIST(
            root = './data',
            train=True, 
            download=True
        )
        self.data_test = td.MNIST(
            root = './data',
            train=False, 
            download=True
        ) 
    
    def info(self): 
        print(self.data_train.classes)
        img, label = self.data_train[0]
        print(img.shape)
        print(img.dtype)
        print(label)

    def train_loader(self): 
        trainLoader = DataLoader(self.data_train, 64, shuffle=True) 
        return trainLoader 
    
    def test_loader(self):
        testLoader = DataLoader(self.data_test, 64) 
        return testLoader 
    

    def normalize_data(self): 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.data_train.transform = transform
        self.data_test.transform = transform

    def convert(self, idx): 
        #Helper function 
        vec = torch.zeros(10) 
        vec[idx] = 1 
        return vec
    def one_hot_encode(self, labels): 
        """
        truyền vào labels lấy từ batch, đó trả ra cái results
        """
        result = []
        for i in labels: 
            result.append(self.convert(i)) 

        return torch.stack(result)

# DataTest = Data() 
# DataTest.normalize_data()
# DataTest.info() 