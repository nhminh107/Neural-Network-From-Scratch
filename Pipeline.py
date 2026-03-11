from Model import Model
from Data import DataLoader
from Data import DataMNIST
import torch 

class Pipeline:
    def __init__(self, data:DataMNIST, model: Model):
        self.data = data
        self.model = model
    
    def train(self, epoch): 
        
        self.data.normalize_data()
        train_loader = self.data.train_loader()

        for i in range(epoch): 
            
            total_loss = 0 
            batch_count = 0 

            for X, y in train_loader:

                X = X.view(X.shape[0], -1) 
                y_onehot = self.data.one_hot_encode(y) 

                cache = self.model.forward(X) 
                y_pred = cache[-1] 

                loss = self.model.Cross_Entropy(y_pred, y_onehot) 

                self.model.updateWeights(X, y_onehot, cache)

                total_loss += loss 
                batch_count += 1 
            print(f"Epoch {i+1}, Loss: {total_loss/batch_count}")
        

    def predict(self):

        test_loader = self.data.test_loader()

        preds = []

        for X, y in test_loader:
            X = X.view(X.shape[0], -1)
            cache = self.model.forward(X)
            probs = cache[-1]
            y_pred = torch.argmax(probs, dim=1)

            preds.append(y_pred)

        return torch.cat(preds)
    
    def accuracy(self):

        test_loader = self.data.test_loader()

        correct = 0
        total = 0

        for X, y in test_loader:

            X = X.view(X.shape[0], -1)

            cache = self.model.forward(X)
            probs = cache[-1]

            pred = torch.argmax(probs, dim=1)

            correct += (pred == y).sum().item()
            total += y.shape[0]

        return correct / total