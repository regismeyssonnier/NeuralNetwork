import time
import random
import numpy as np

class NeuralNet:

    def __init__(self, sz_inp, size, outsz, lr):

        self.size_input = sz_inp
        self.size_layer = size
        self.size_output = outsz
        self.layer1 = np.zeros(size, dtype=np.float)
        self.layero = np.zeros(outsz, dtype=np.float)
        self.layer1w = (np.random.random((self.size_input, self.size_layer))*40.0+1)/100.0
        self.layerow = (np.random.random((self.size_layer, self.size_output))*40.0+1)/100.0
        self.layer1b = np.zeros(size)
        self.layerob = np.zeros(outsz)
        self.LR = lr

    def Relu(self, x):
        return np.maximum(0, x)

    def dRELU(self, x):
        return np.greater(x, 0)

    def build(self, inp):
        self.input = inp

    def forward(self):

        self.layer1 = self.Relu(np.dot(self.input, self.layer1w) + self.layer1b)
        self.layero = self.Relu(np.dot(self.layer1, self.layerow) + self.layerob)

    def backward(self, target):
      
        cost = 2 * (self.layero - target)
        
        for o in range(self.size_output):
            for i in range(self.size_layer):
                self.layerow[i][o] -= self.LR *\
                   cost[o] *\
                  self.layer1[i]

        for o in range(self.size_output):
            self.layerob[o] -=  self.LR * cost[o]
       

        for o in range(self.size_output):
            for f in range(self.size_layer):
                for i in range(self.size_input):
                        self.layer1w[i][f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f]) * self.input[i]

        for o in range(self.size_output):
            for f in range(self.size_layer):
                self.layer1b[f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f])
               
    def backward2(self, target):
      
        cost = 2 * (self.layero - target)
        
        for i in range(self.size_input):
            for f in range(self.size_layer):
                for o in range(self.size_output):    
                    self.layerow[i][o] -= self.LR * cost[o] * self.layer1[i]
                    self.layerob[o] -=  self.LR * cost[o]
                    self.layer1w[i][f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f]) * self.input[i]
                    self.layer1b[f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f])
        

    def predict(self, inp):
        self.layer1 = self.Relu(np.dot(inp, self.layer1w) + self.layer1b)
        self.layero = self.Relu(np.dot(self.layer1, self.layerow) + self.layerob)

        for i in range(self.size_output):
            print("out 1: " + str(self.layero[i]))

nn = NeuralNet(2, 5, 1, 0.00001)
for i in range(100):
    for j in range(100):
        nn.build(np.array([i, j]))
        nn.forward()
        nn.backward2(np.array([i+j]))
nn.predict(np.array([200, 75]))