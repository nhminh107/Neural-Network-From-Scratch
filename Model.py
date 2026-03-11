import torch


class Model: 
    def __init__(self):
        """Cho architecture là 784 -> 128 -> 64 -> 10""" 
        self.params = {} 
    
    def init_params(self): 
        self.params["W1"] = torch.randn(784, 128)*0.01
        self.params["b1"] = torch.zeros(128) 
        self.params["W2"] = torch.randn(128, 64)*0.01
        self.params["b2"] = torch.zeros(64)
        self.params["W3"] = torch.randn(64, 10) * 0.01 
        self.params["b3"] = torch.zeros(10)

    def relu(self, x): 
        return torch.maximum(torch.zeros_like(x), x) 
    
    def softmax(self,x):
        exp = torch.exp(x - torch.max(x,dim=1,keepdim=True).values)
        return exp / torch.sum(exp,dim=1,keepdim=True)
        
    def Cross_Entropy(self, y_pred, y_true): 
        epsilon = 1e-15
        y_pred = torch.clip(y_pred, epsilon, 1 - epsilon)

        loss = -torch.sum(y_true*torch.log(y_pred)) 
        return loss / y_true.shape[0] 
    
    def forward(self, X): 
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]
        W3 = self.params["W3"]
        b3 = self.params["b3"]

        z1 = X @ W1 + b1 
        a1 = self.relu(z1) 

        z2 = a1 @ W2 + b2 
        a2 = self.relu(z2)

        z3 = a2 @ W3 + b3 
        out = self.softmax(z3) 

        cache = [z1, a1, z2, a2, z3, out]
        return cache  
    
    def backward(self, X, y, cache):
        W1 = self.params["W1"]
        W2 = self.params["W2"] 
        W3 = self.params["W3"] 

        z1, a1, z2, a2, z3,out = cache

        m = X.shape[0]
        grads = {} 

        #Layer 3 
        """gradient = input^T * error"""
        dz3 = out - y 
        grads["W3"] = (a2.T @ dz3)/m
        grads["b3"] = torch.sum(dz3, dim = 0)/m
        #Layer 2 
        """dz3 lớn thì layer 2 phải chịu trách nhiệm 
        da2 = dL/d(A2) = dL/d(z3) * d(z2)/d(a2). CHAIN RULE
        Điều đó cũng tương tự với dz2"""
        da2 = dz3 @ W3.T 
        dz2 = da2* (z2 > 0) 

        grads["W2"] = (a1.T @ dz2)/m
        grads["b2"] = torch.sum(dz2, dim=0)/m 

        #Layer1 
        
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0) 

        grads["W1"] = (X.T @ dz1)/m
        grads["b1"] = torch.sum(dz1, dim = 0)/m

        return grads


    def updateWeights(self, X, y, cache): 
        grads = self.backward(X, y, cache) 

        lr = 0.09
        for i in range (1, 4):
            W_i = "W" + str(i) 
            b_i = "b" + str(i) 

            self.params[W_i] = self.params[W_i] - lr*grads[W_i]
            self.params[b_i] = self.params[b_i] - lr*grads[b_i]
            
    
