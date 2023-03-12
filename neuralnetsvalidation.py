import time
import random
import numpy as np
from nnp import *
from im import *
from save import *
import time
import random

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0



    def update(self, w, grad_wrt_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_wrt_w)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w


class ConvNeuralNet:

    def __init__(self, sz_inp, size, outsz, lr):
        self.cost = np.array([0.0, 0.0, 0.0])
        self.size_input = sz_inp
        self.size_layer = size
        self.size_output = outsz
        self.layer1 = np.zeros(size, dtype=np.float)
        self.layero = np.zeros(outsz, dtype=np.float)
        self.loss =0# np.array([0.0, 0.0, 0.0, 0.0])
        self.dropout_mask = 0
        self.dropout_mask2 = 0
        self.target_m = np.array([0.0, 0.0, 0.0])
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.f1score = 0

        """
        self.layer1w = (np.random.random((self.size_input, self.size_layer))*40.0+1)/100.0
        self.layerow = (np.random.random((self.size_layer, self.size_output))*40.0+1)/100.0
        self.layer1b = (np.random.random(size)*40.0+1)/100.0
        self.layerob = (np.random.random(outsz)*40.0+1)/100.0"""

        self.layer1w = np.random.randn(self.size_input, self.size_layer) / np.sqrt(self.size_input)
        self.layerow = np.random.randn(self.size_layer, self.size_output) / np.sqrt(self.size_layer)
        self.layer1b = 0.1
        self.layerob = 0.1

        self.layer1_s = np.zeros(self.size_layer)
        self.layer1w_s = np.zeros((self.size_input, self.size_layer))
        self.layer1b_s = 0

        self.layero_s = np.zeros(self.size_output)
        self.layerow_s = np.zeros((self.size_layer, self.size_output))
        self.layerob_s = 0

        """
        self.layer1w = np.random.randn(self.size_input, self.size_layer) * np.sqrt(2 / self.size_input)
        self.layerow = np.random.randn(self.size_layer, self.size_output) * np.sqrt(2 / self.size_layer)
        self.layer1b = np.zeros(self.size_layer)
        self.layerob = np.zeros(self.size_output)
        """

        self.LR = lr
        self.aug_dimg = []
        self.aug_dimg_cl = []
        self.aug_dimg2 = []
        self.aug_dimg_cl2 = []

    def add_layer2(self, size):
        self.size_layer2 = size
        self.layer2w = self.layer2w = np.random.randn(self.size_layer, self.size_layer2) / np.sqrt(self.size_layer + self.size_layer2)
        self.layer2b = 0.1

        self.layer1w = np.random.randn(self.size_input, self.size_layer) / np.sqrt(self.size_input)
        self.layerow = np.random.randn(self.size_layer2, self.size_output) / np.sqrt(self.size_layer2)

        
        self.layer1w_s = np.zeros((self.size_input, self.size_layer))
        self.layerow_s = np.zeros((self.size_layer2, self.size_output))
        
        self.layer2_s = np.zeros(self.size_layer2)
        self.layer2w_s = np.zeros((self.size_layer, self.size_layer2))
        self.layer2b_s = 0.1
        
    def save_weight_bias(self):
        for i in range(self.size_output):
            self.layero_s[i] = self.layero[i]

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow_s[j][i] = self.layerow[j][i]

        self.layerob_s = self.layerob

        for i in range(self.size_layer):
            self.layer1_s[i] = self.layer1[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w_s[i][f] = self.layer1w[i][f]

        self.layer1b_s = self.layer1b

    def save_weight_bias2(self):
        for i in range(self.size_output):
            self.layero_s[i] = self.layero[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow_s[j][i] = self.layerow[j][i]

        self.layerob_s = self.layerob

        for i in range(self.size_layer2):
            self.layer2_s[i] = self.layer2[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w_s[i][f] = self.layer2w[i][f]

        self.layer2b_s = self.layer2b

        for i in range(self.size_layer):
            self.layer1_s[i] = self.layer1[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w_s[i][f] = self.layer1w[i][f]

        self.layer1b_s = self.layer1b

    def save_file(self, name):
        f = open(name, "w")
       
        for i in range(self.size_output):
            f.write(str(self.layero_s[i]) + "\n")

        for i in range(self.size_output):
            for j in range(self.size_layer):
                f.write(str(self.layerow_s[j][i])+ "\n")

        f.write(str(self.layerob_s) + "\n")

        for i in range(self.size_layer):
            f.write(str(self.layer1_s[i])+ "\n")

        for l in range(self.size_layer):
            for i in range(self.size_input):
                f.write(str(self.layer1w_s[i][l])+ "\n")

        f.write(str(self.layer1b_s)+ "\n")

        f.close()

    def save_file2(self, name):
        f = open(name, "w")
       
        for i in range(self.size_output):
            f.write(str(self.layero_s[i]) + "\n")

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                f.write(str(self.layerow_s[j][i])+ "\n")

        f.write(str(self.layerob_s) + "\n")

        for i in range(self.size_layer2):
            f.write(str(self.layer2_s[i])+ "\n")

        for l in range(self.size_layer2):
            for i in range(self.size_layer):
                f.write(str(self.layer2w_s[i][l])+ "\n")

        f.write(str(self.layer2b_s)+ "\n")

        for i in range(self.size_layer):
            f.write(str(self.layer1_s[i])+ "\n")

        for l in range(self.size_layer):
            for i in range(self.size_input):
                f.write(str(self.layer1w_s[i][l])+ "\n")

        f.write(str(self.layer1b_s)+ "\n")

        f.close()

    def load_file(self, name):
        f = open(name, "r")
       
        for i in range(self.size_output):
            self.layero[i] = float(f.readline())

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow[j][i] = float(f.readline())

        self.layerob = float(f.readline())

        for i in range(self.size_layer):
            self.layer1=float(f.readline())

        for l in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][l] = float(f.readline())

        self.layer1b = float(f.readline())

        f.close()

    def load_file2(self, name):
        f = open(name, "r")
       
        for i in range(self.size_output):
            self.layero[i] = float(f.readline())

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow[j][i] = float(f.readline())

        self.layerob = float(f.readline())

        for i in range(self.size_layer2):
            self.layer2=float(f.readline())

        for l in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w[i][l] = float(f.readline())

        self.layer2b = float(f.readline())

        for i in range(self.size_layer):
            self.layer1=float(f.readline())

        for l in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][l] = float(f.readline())

        self.layer1b = float(f.readline())

        f.close()

    def load_weight_bias(self):
        for i in range(self.size_output):
            self.layero[i] = self.layero_s[i]

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow[j][i] = self.layerow_s[j][i]

        self.layerob = self.layerob_s

        for i in range(self.size_layer):
            self.layer1[i] = self.layer1_s[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][f] = self.layer1w_s[i][f]

        self.layer1b = self.layer1b_s

    def load_weight_bias2(self):
        for i in range(self.size_output):
            self.layero[i] = self.layero_s[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow[j][i] = self.layerow_s[j][i]

        self.layerob = self.layerob_s

        for i in range(self.size_layer2):
            self.layer2[i] = self.layer2_s[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w[i][f] = self.layer2w_s[i][f]

        self.layer2b = self.layer2b_s

        for i in range(self.size_layer):
            self.layer1[i] = self.layer1_s[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][f] = self.layer1w_s[i][f]

        self.layer1b = self.layer1b_s


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def derive(self, x):
	    return x * (1 - x)

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha*x, x)
    def dleaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def Relu(self, x):
        return np.maximum(0, x)

    def dRELU(self, x):
        return np.greater(x, 0)

    def build(self, inp):
        self.input = inp

    def schedule_lr(self, lr_decay, iteration):
        return self.LR / (1.0 + lr_decay * iteration)


    def BN(self, scale):

        s = 0
        for i in range(self.size_input):
	        s += self.input[i]

        s /= self.size_input

        sx = 0
        d = 0
        for i in range(self.size_input):
	        d += (self.input[i] - s)*(self.input[i] - s)
		
        d /= self.size_input
	
        Y = np.zeros(self.size_input)
        I = 0
        for i in range(self.size_input):
	        Y[I] =  ((self.input[i] - s) / np.sqrt(d + 0.000001)) * scale 
	        I+=1
        self.input = Y

    def augmentation_data(self, nb, cl):
        
        for i in range(nb):
            img = []
            for j in range(self.size_input):
                img.append(self.input[i] * (random.random()+0.000001))
            self.aug_dimg.append(img)
            self.aug_dimg_cl.append(cl)

    def augmentation_data2(self, nb, cl):
        
        for i in range(nb):
            img = []
            for j in range(self.size_input):
                img.append(self.input[i] * (random.random()+0.000001))
            self.aug_dimg2.append(img)
            self.aug_dimg_cl2.append(cl)

    def get_L2(self):
        sum = 0
        n= 0
        for o in range(self.size_output):
            for i in range(self.size_layer):
                sum += self.layerow[i][o] ** 2
                n+=1

        for f in range(self.size_layer):
            for i in range(self.size_input):
                sum += self.layer1w[i][f] ** 2
                n+=1

        return sum * (0.00001 / (2.0 * n))

    def forward(self):

        self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

  

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def predict_softmax2(self):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        self.layer2 = self.leaky_relu(np.dot(self.layer1, self.layer2w) + self.layer2b)
        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer2, self.layerow + self.layerob))

    def predict_softmax(self):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
       
        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer1, self.layerow + self.layerob))

    def forward_drop(self, target, nb, ndrop):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        hidden_output = self.leaky_relu(np.dot(self.input, self.layer1w)+ self.layer1b) 
        if ndrop:
            self.dropout_mask = np.random.binomial(1, 0.2, size=hidden_output.shape)

        #print(dropout_mask)
        self.layer1 = hidden_output * self.dropout_mask / 0.2

        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer1, self.layerow) + self.layerob)
        self.cost += (self.layero - target)
        max = -float('inf')
        ind = 0
        for i in range(self.size_output):
            if self.layero[i] > max:
                max = self.layero[i]
                ind = i
        for i in range(self.size_output):
            if target[i] == 1.0 and ind == i:
                self.loss+=1

    def forward_drop2(self, target, nb, ndrop):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        hidden_output = self.leaky_relu(np.dot(self.input, self.layer1w)+ self.layer1b) 
        if ndrop:
            self.dropout_mask = np.random.binomial(1, 0.2, size=hidden_output.shape)

        #print(dropout_mask)
        self.layer1 = hidden_output * self.dropout_mask / 0.2

        hidden_output2 = self.leaky_relu(np.dot(self.layer1, self.layer2w)+ self.layer2b) 
        if ndrop:
            self.dropout_mask2 = np.random.binomial(1, 0.2, size=hidden_output2.shape)

        #print(dropout_mask)
        self.layer2 = hidden_output2 * self.dropout_mask2 / 0.2


        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer2, self.layerow) + self.layerob)



        self.cost += (self.layero - target)
        max = -float('inf')
        ind = 0
        for i in range(self.size_output):
            if self.layero[i] > max:
                max = self.layero[i]
                ind = i

        for i in range(self.size_output):
            if target[i] == 1.0:
                self.target_m[i] += 1
            if target[i] == 1.0 and ind == i:
           
                self.loss+=1
      
           

    def backward_np(self, Y, batch_sz):
        dYp = (self.layero - Y) / batch_sz
        dlayerow = np.dot(self.layer1.T, dYp) + 2 * 2 * self.layerow
        dhidden_output = np.dot(dYp, self.layerow.T) * self.dleaky_relu(hidden_output)
        dlayer1w = np.dot(self.input.T, dhidden_output) + 2 * 2 * self.layer1w
        self.layer1w -= self.LR * dlayer1w
        self.layerow -= self.LR * dlayerow


    def backward(self, batch_sz):
        
        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        self.cost = self.cost / batch_sz + reg
        #cost =  (self.layero - target) / batch_sz
        #print("cost " + str(cost))
        #cost = (-target / self.layero) / batch_sz

        
        for o in range(self.size_output):
            for i in range(self.size_layer):
                dw = (self.cost[o] * self.layer1[i]) + 0.01* self.layerow[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow[i][o], dw)
                """if self.layerow[i][o] > 0.06:
                    self.layerow[i][o] = 0.06
                if self.layerow[i][o] < -0.06:
                    self.layerow[i][o] = -0.06"""
                #print(str(self.layerow[i][o]) + "  " + str(self.LR * (cost[o] * self.layer1[i])))
        db = 0
        for o in range(self.size_output):
            db += self.LR * self.cost[o]
        self.layerob -= db

        for o in range(self.size_output):
            for f in range(self.size_layer):
                for i in range(self.size_input):
                        dw = (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])\
                            + 0.01 * self.layer1w[i][f]
                        adam = Adam(learning_rate = self.LR)
                        self.layer1w[i][f] = adam.update(self.layer1w[i][f], dw)
                        if self.layer1w[i][f] > 0.006:
                            self.layer1w[i][f] = 0.006
                        if self.layer1w[i][f] < -0.006:
                            self.layer1w[i][f] = -0.006
                        #if 0.001 >= self.layer1w[i][f] >= -0.001:
                         #   self.layer1w[i][f] = 0.002

                        #print(str(self.layer1w[i][f]) + " " + str(self.LR * (cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])))
        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer):
                db+= self.LR * self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f])
        self.layer1b -= db

        #self.reinforcment()

        self.cost = np.array([0.0, 0.0, 0.0])

    
    def backward2(self, batch_sz):
        
        reg = 0.001 * (np.sum(np.square(self.layer1w))+ np.sum(np.square(self.layer2w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        self.cost = self.cost / batch_sz + reg
        #cost =  (self.layero - target) / batch_sz
        #print("cost " + str(cost))
        #cost = (-target / self.layero) / batch_sz

        
        for o in range(self.size_output):
            for i in range(self.size_layer2):
                dw = (self.cost[o] * self.layer2[i]) + 0.001* self.layerow[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow[i][o], dw)
                """if self.layerow[i][o] > 0.06:
                    self.layerow[i][o] = 0.06
                if self.layerow[i][o] < -0.06:
                    self.layerow[i][o] = -0.06"""
                #print(str(self.layerow[i][o]) + "  " + str(self.LR * (cost[o] * self.layer1[i])))
        db = 0
        for o in range(self.size_output):
            db += self.LR * self.cost[o]
        self.layerob -= db

        for o in range(self.size_output):
            for f in range(self.size_layer2):
                for i in range(self.size_layer):
                        dw = (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f]) * self.layer1[i])\
                            + 0.001 * self.layer2w[i][f]
                        adam = Adam(learning_rate = self.LR)
                        self.layer2w[i][f] = adam.update(self.layer2w[i][f], dw)
                        """if self.layer2w[i][f] > 0.006:
                            self.layer2w[i][f] = 0.006
                        if self.layer2w[i][f] < -0.006:
                            self.layer2w[i][f] = -0.006"""
                        #if 0.001 >= self.layer1w[i][f] >= -0.001:
                         #   self.layer1w[i][f] = 0.002

                        #print(str(self.layer1w[i][f]) + " " + str(self.LR * (cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])))
        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                db+= self.LR * self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f])
        self.layer2b -= db

        for o in range(self.size_output):
            for f in range(self.size_layer2):
                for i in range(self.size_layer):
                    for j in range(self.size_input):
                        dw = (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f]) * self.layer2w[i][f] * \
                              self.dleaky_relu(self.layer1[i]) * self.input[j])\
                            + 0.001 * self.layer1w[j][i]
                        adam = Adam(learning_rate = self.LR)
                        self.layer1w[j][i] = adam.update(self.layer1w[j][i], dw)
                        """if self.layer1w[j][i] > 0.006:
                            self.layer1w[j][i] = 0.006
                        if self.layer1w[j][i] < -0.006:
                            self.layer1w[j][i] = -0.006"""
                        #if 0.001 >= self.layer1w[i][f] >= -0.001:
                         #   self.layer1w[i][f] = 0.002

                        #print(str(self.layer1w[i][f]) + " " + str(self.LR * (cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])))
        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                 for i in range(self.size_layer):
                    db+= (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f]) * self.layer2w[i][f] \
                      * self.dleaky_relu(self.layer1[i]))

        self.layer1b -= db

        self.cost = np.array([0.0, 0.0, 0.0])
        self.target_m = np.array([0.0, 0.0, 0.0])
        self.tp = 0
        self.fp = 0
        self.fn=  0

    def backward_tuning(self, batch_sz):
        
        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
        #cost = (self.layero - target) / batch_sz + reg
        self.cost = self.cost / batch_sz + reg
        #cost =  (self.layero - target) / batch_sz
        #print("cost " + str(cost))
        #cost = (-target / self.layero) / batch_sz

        
        for o in range(self.size_output):
            for i in range(self.size_layer):
                dw = (self.cost[o] * self.layer1[i]) + 0.01* self.layerow[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow[i][o], dw)
               
                #print(str(self.layerow[i][o]) + "  " + str(self.LR * (cost[o] * self.layer1[i])))
        db = 0
        for o in range(self.size_output):
            db += self.LR * self.cost[o]
        self.layerob -= db


        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def reinforcment(self):
               
        for f in range(0, self.size_layer, 2):
            for i in range(self.size_input):
                if 0.001 >= self.layer1w[i][f] >= -0.001:
                    self.layer1w[i][f] = 0.006

    def backward2f(self, target):
      
        cost = 2 * (self.layero - target)
        
        for i in range(self.size_input):
            for f in range(self.size_layer):
                for o in range(self.size_output):    
                    self.layerow[i][o] -= self.LR * cost[o] * self.layer1[i]
                    self.layerob[o] -=  self.LR * cost[o]
                    self.layer1w[i][f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f]) * self.input[i]
                    self.layer1b[f] -= self.LR * cost[o] * self.layerow[f][o] * self.dRELU(self.layer1[f])
        
    def change_lr(self):
        self.LR = random.randint(10, 20) / 100.0

    def set_lr(self, lr):
        self.LR = lr

    def predict(self, inp):
        self.layer1 = self.Relu(np.dot(inp, self.layer1w))# + self.layer1b)
        self.layero = self.Relu(np.dot(self.layer1, self.layerow))# + self.layerob)

        for i in range(self.size_output):
            print("out 1: " + str(self.layero[i]))

    def F1_score(self):
        prec= reca=0
        if self.tp + self.fp == 0.0:prec = 0.0
        else: prec = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0.0:reca = 0.0
        else: reca = self.tp / (self.tp + self.fn)
        self.f1score = 0.0
        if prec+reca != 0.0: self.f1score = 2 * (prec * reca) / (prec + reca)
        return self.f1score

    def init_F1(self):
        self.tp = self.fn = self.fp = 0.0



"""nn = ConvNeuralNet(2, 5, 1, 0.00001)
for i in range(100):
    for j in range(100):
        nn.build(np.array([i, j]))
        nn.forward()
        nn.backward(np.array([i+j]))
nn.predict(np.array([200, 75]))"""

nn = ConvNeuralNet(256, 32, 4, 0.001)

NB_IMAGE = 192
start_time = time.time()


TESTV, TESTV_out, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3 = create_test_tab_one_filter_rand_batch_train3(64)

nn.load_file("nnmodel/56.77_256x32x4x0.001_1.txt")

lotsz = 32
alt = 1
ilot = 0

res = [0,0,0]
valid_tot = np.zeros(NB_IMAGE)
valid = np.zeros(NB_IMAGE)
score, pct = ValidationNN(nn, TESTV, TESTV_out, 0.1, valid, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3, res)
for v in range(NB_IMAGE):
    if valid[v]:
        valid_tot[v] = 1

for i in range(3):
    print(res[i])
    pc = res[i] / NB_IMAGE
    print(str(pc))   

PATHF = []
ilot = 0
alt = 1
I=I2=I3=0
for i in range(NB_IMAGE):
    if alt == 1:
        PATHF.append(PATHV[I])
        I+=1
    elif alt == 2:
        PATHF.append(PATHV2[I2])
        I2+=1
    else:
        PATHF.append(PATHV3[I3])
        I3+=1

    ilot+=1
    if ilot == lotsz:
        ilot = 0
        alt+=1
        if alt == 4:
            alt = 1



ct = 0
for v in range(NB_IMAGE):
    if valid_tot[v]:
        ct += 1
    
pct = ct / NB_IMAGE * 100.0
print("---------------------------====-------------------------------")
print("Totale :" + str(pct) + "% image correctement classe")

duration = time.time() - start_time 
print("duree : " + str(duration) + " s")

Afficher_images2(PATHF, valid_tot, NB_IMAGE, score, pct)

print(nn.layero)
#print(nn.layerow)
print(nn.layerob)