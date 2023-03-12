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
        self.cost = np.array([0.0, 0.0, 0.0, 0.0])
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
        self.layer1b = 0.1 * np.random.randn(self.size_layer)
        self.layerob = 0.1 * np.random.randn(self.size_output)

        self.layer1_s = np.zeros(self.size_layer)
        self.layer1w_s = np.zeros((self.size_input, self.size_layer))
        self.layer1b_s = np.zeros(self.size_layer)

        self.layero_s = np.zeros(self.size_output)
        self.layerow_s = np.zeros((self.size_layer, self.size_output))
        self.layerob_s = np.zeros(self.size_output)

        self.layer1_s2 = np.zeros(self.size_layer)
        self.layer1w_s2 = np.zeros((self.size_input, self.size_layer))
        self.layer1b_s2 = np.zeros(self.size_layer)

        self.layero_s2 = np.zeros(self.size_output)
        self.layerow_s2 = np.zeros((self.size_layer, self.size_output))
        self.layerob_s2 = np.zeros(self.size_output)

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
        self.layer2 = np.zeros(size, dtype=np.float)
        self.layer2w = self.layer2w = np.random.randn(self.size_layer, self.size_layer2) / np.sqrt(self.size_layer + self.size_layer2)
        self.layer2b = 0.1

        self.layer1w = np.random.randn(self.size_input, self.size_layer) / np.sqrt(self.size_input)
        self.layerow = np.random.randn(self.size_layer2, self.size_output) / np.sqrt(self.size_layer2)

        
        self.layer1w_s = np.zeros((self.size_input, self.size_layer))
        self.layerow_s = np.zeros((self.size_layer2, self.size_output))
        
        self.layer2_s = np.zeros(self.size_layer2)
        self.layer2w_s = np.zeros((self.size_layer, self.size_layer2))
        self.layer2b_s = np.zeros(self.size_layer2)

        
        self.layer1w_s2 = np.zeros((self.size_input, self.size_layer))
        self.layerow_s2 = np.zeros((self.size_layer2, self.size_output))
        
        self.layer2_s2 = np.zeros(self.size_layer2)
        self.layer2w_s2 = np.zeros((self.size_layer, self.size_layer2))
        self.layer2b_s2 = np.zeros(self.size_layer2)
        
    def save_weight_bias(self):
        for i in range(self.size_output):
            self.layero_s[i] = self.layero[i]
            self.layerob_s[i] = self.layerob[i]

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow_s[j][i] = self.layerow[j][i]


        

        for i in range(self.size_layer):
            self.layer1_s[i] = self.layer1[i]
            self.layer1b_s[i] = self.layer1b[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w_s[i][f] = self.layer1w[i][f]

        

    def save_weight_bias2(self):
        for i in range(self.size_output):
            self.layero_s[i] = self.layero[i]
            self.layerob_s[i] = self.layerob[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow_s[j][i] = self.layerow[j][i]

        

        for i in range(self.size_layer2):
            self.layer2_s[i] = self.layer2[i]
            self.layer2b_s[i] = self.layer2b[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w_s[i][f] = self.layer2w[i][f]

        

        for i in range(self.size_layer):
            self.layer1_s[i] = self.layer1[i]
            self.layer1b_s[i] = self.layer1b[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w_s[i][f] = self.layer1w[i][f]

        

    def save_weight_bias22(self):
        for i in range(self.size_output):
            self.layero_s2[i] = self.layero[i]
            self.layerob_s2[i] = self.layerob[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow_s2[j][i] = self.layerow[j][i]

        

        for i in range(self.size_layer2):
            self.layer2_s2[i] = self.layer2[i]
            self.layer2b_s2[i] = self.layer2b[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w_s2[i][f] = self.layer2w[i][f]

        

        for i in range(self.size_layer):
            self.layer1_s2[i] = self.layer1[i]
            self.layer1b_s2[i] = self.layer1b[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w_s2[i][f] = self.layer1w[i][f]

        

    def save_file(self, name):
        f = open(name, "w")
       
        for i in range(self.size_output):
            f.write(str(self.layero_s[i]) + "\n")

        for i in range(self.size_output):
            for j in range(self.size_layer):
                f.write(str(self.layerow_s[j][i])+ "\n")

        for i in range(self.size_output):
            f.write(str(self.layerob_s[i]) + "\n")

        for i in range(self.size_layer):
            f.write(str(self.layer1_s[i])+ "\n")

        for l in range(self.size_layer):
            for i in range(self.size_input):
                f.write(str(self.layer1w_s[i][l])+ "\n")

        for l in range(self.size_layer):
            f.write(str(self.layer1b_s[l])+ "\n")

        f.close()

    def save_file2(self, name):
        f = open(name, "w")
       
        for i in range(self.size_output):
            f.write(str(self.layero_s[i]) + "\n")

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                f.write(str(self.layerow_s[j][i])+ "\n")

        for i in range(self.size_output):
            f.write(str(self.layerob_s[i]) + "\n")

        for i in range(self.size_layer2):
            f.write(str(self.layer2_s[i])+ "\n")

        for l in range(self.size_layer2):
            for i in range(self.size_layer):
                f.write(str(self.layer2w_s[i][l])+ "\n")

        for l in range(self.size_layer2):
            f.write(str(self.layer2b_s[l])+ "\n")

        for i in range(self.size_layer):
            f.write(str(self.layer1_s[i])+ "\n")

        for l in range(self.size_layer):
            for i in range(self.size_input):
                f.write(str(self.layer1w_s[i][l])+ "\n")

        for l in range(self.size_layer):
            f.write(str(self.layer1b_s[l])+ "\n")

        f.close()

    def load_file(self, name):
        f = open(name, "r")
       
        for i in range(self.size_output):
            self.layero[i] = float(f.readline())

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow[j][i] = float(f.readline())

        for i in range(self.size_output):
            self.layerob[i] = float(f.readline())

        for i in range(self.size_layer):
            self.layer1=float(f.readline())

        for l in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][l] = float(f.readline())

        for l in range(self.size_layer):
            self.layer1b[l] = float(f.readline())

        f.close()

    def load_file2(self, name):
        f = open(name, "r")
       
        for i in range(self.size_output):
            self.layero[i] = float(f.readline())

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow[j][i] = float(f.readline())

        for i in range(self.size_output):
            self.layerob[i] = float(f.readline())

        for i in range(self.size_layer2):
            self.layer2=float(f.readline())

        for l in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w[i][l] = float(f.readline())

        for l in range(self.size_layer2):
            self.layer2b[l] = float(f.readline())

        for i in range(self.size_layer):
            self.layer1=float(f.readline())

        for l in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][l] = float(f.readline())

        for l in range(self.size_layer):
            self.layer1b[l] = float(f.readline())

        f.close()

    def load_weight_bias(self):
        for i in range(self.size_output):
            self.layero[i] = self.layero_s[i]
            self.layerob[i] = self.layerob_s[i]

        for i in range(self.size_output):
            for j in range(self.size_layer):
                self.layerow[j][i] = self.layerow_s[j][i]
                  

        for i in range(self.size_layer):
            self.layer1[i] = self.layer1_s[i]
            self.layer1b[i] = self.layer1b_s[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][f] = self.layer1w_s[i][f]

        

    def load_weight_bias2(self):
        for i in range(self.size_output):
            self.layero[i] = self.layero_s[i]
            self.layerob[i] = self.layerob_s[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow[j][i] = self.layerow_s[j][i]

        

        for i in range(self.size_layer2):
            self.layer2[i] = self.layer2_s[i]
            self.layer2b[i] = self.layer2b_s[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w[i][f] = self.layer2w_s[i][f]

        

        for i in range(self.size_layer):
            self.layer1[i] = self.layer1_s[i]
            self.layer1b[i] = self.layer1b_s[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][f] = self.layer1w_s[i][f]

        

    def load_weight_bias22(self):
        for i in range(self.size_output):
            self.layero[i] = self.layero_s2[i]
            self.layerob[i] = self.layerob_s2[i]

        for i in range(self.size_output):
            for j in range(self.size_layer2):
                self.layerow[j][i] = self.layerow_s2[j][i]

        

        for i in range(self.size_layer2):
            self.layer2[i] = self.layer2_s2[i]
            self.layer2b[i] = self.layer2b_s2[i]

        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                self.layer2w[i][f] = self.layer2w_s2[i][f]

        

        for i in range(self.size_layer):
            self.layer1[i] = self.layer1_s2[i]
            self.layer1b[i] = self.layer1b_s2[i]

        for f in range(self.size_layer):
            for i in range(self.size_input):
                self.layer1w[i][f] = self.layer1w_s2[i][f]

        


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

    def optimInput(self):

        m = 0
        for i in range(self.size_input):
	        m += self.input[i]

        m /= self.size_input
        for i in range(self.size_input):
            self.input[i] -= m

        v = 0
        for i in range(self.size_input):
            v+= self.input[i]**2
        v /= self.size_input

        for i in range(self.size_input):
            self.input[i] /= v



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

    def softmax_deriv(self, x):
        """ Dérivée de la fonction softmax """
        return np.diag(x) - np.outer(x, x)

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
            drop = [0]*nb
            drop += [1]*(self.size_layer-nb)
            #self.dropout_mask = np.random.binomial(1, 0.7, size=hidden_output.shape)
            random.shuffle(drop)
            self.dropout_mask = np.array(drop)
        #print(dropout_mask)
        self.layer1 = hidden_output * self.dropout_mask #/ 0.6

        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer1, self.layerow) + self.layerob)
        self.cost += (self.layero - target) 
        #self.cost += np.dot(self.softmax_deriv(self.layero), (self.layero - target).T)

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
            drop = [0]*nb
            drop += [1]*(self.size_layer-nb)
            #self.dropout_mask = np.random.binomial(1, 0.7, size=hidden_output.shape)
            random.shuffle(drop)
            self.dropout_mask = np.array(drop)
        #print(dropout_mask)
        self.layer1 = hidden_output * self.dropout_mask

        hidden_output2 = self.leaky_relu(np.dot(self.layer1, self.layer2w)+ self.layer2b) 
        if ndrop:
            drop = [0]*nb
            drop += [1]*(self.size_layer2-nb)
            #self.dropout_mask = np.random.binomial(1, 0.7, size=hidden_output.shape)
            random.shuffle(drop)
            self.dropout_mask2 = np.array(drop)
        #print(dropout_mask)
        self.layer2 = hidden_output2 * self.dropout_mask2


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
        self.layerob -= db / self.size_output
        
        for o in range(self.size_output):
            for f in range(self.size_layer):
                for i in range(self.size_input):
                        dw = (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])\
                            + 0.01 * self.layer1w[i][f]
                        adam = Adam(learning_rate = self.LR)
                        self.layer1w[i][f] = adam.update(self.layer1w[i][f], dw)
                        """if self.layer1w[i][f] > 0.006:
                            self.layer1w[i][f] = 0.006
                        if self.layer1w[i][f] < -0.006:
                            self.layer1w[i][f] = -0.006"""
                        #if 0.001 >= self.layer1w[i][f] >= -0.001:
                         #   self.layer1w[i][f] = 0.002

                        #print(str(self.layer1w[i][f]) + " " + str(self.LR * (cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f]) * self.input[i])))
        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer):
                db+= self.LR * self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f])
        self.layer1b -= db / (self.size_output* self.size_layer)


        #self.reinforcment()

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backwardl(self, batch_sz):
        
        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        self.cost = (self.cost / batch_sz) + reg
                
        for o in range(self.size_output):
            for i in range(self.size_layer):
                dw = (self.cost[o] * self.layer1[i])
                dw += 0.01* self.layerow[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow[i][o], dw)
               
        db = 0
        for o in range(self.size_output):
            self.layerob[o] -= self.LR * self.cost[o]
         
        
        for i in range(self.size_input):
            for f in range(self.size_layer):
                dw = 0
                for o in range(self.size_output):
                    dw += (self.cost[o] * self.layerow[f][o]) 
                #dw /= self.size_output
                dw *=(self.dleaky_relu(self.layer1[f]) * self.input[i])
                dw += 0.01 * self.layer1w[i][f]
                adam = Adam(learning_rate = self.LR)
                self.layer1w[i][f] = adam.update(self.layer1w[i][f], dw)
                       
        
        for f in range(self.size_layer):
            db = 0
            for o in range(self.size_output):
                db += self.cost[o] * self.layerow[f][o] 
            #db /= self.size_output
            db *= self.dleaky_relu(self.layer1[f])
            self.layer1b -= self.LR * db

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backward_youtube(self, batch_sz):

        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        error = (self.cost / batch_sz) + reg
        self.layerob -= self.LR * error
        dw = np.outer(error, self.layer1).T + 0.01 * self.layerow
        adam = Adam(learning_rate = self.LR)
        self.layerow = adam.update(self.layerow, dw)

        error = np.dot(self.layerow, error) * self.dleaky_relu(self.layer1)
        self.layer1b -= self.LR * error
        dw2 = np.outer(error, self.input).T+ 0.01 * self.layer1w
        adam = Adam(learning_rate = self.LR)
        self.layer1w = adam.update(self.layer1w, dw2)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    


    def backward_matrix(self, batch_sz):
        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        self.cost = self.cost / batch_sz + reg

        dw = np.outer(self.layer1, self.cost)
        dw = dw + 0.01 * self.layerow
        adam = Adam(learning_rate=self.LR)
        self.layerow = adam.update(self.layerow, dw)

        db = np.sum(self.cost) * self.LR
        self.layerob -= db #/ self.size_output

        
        for i in range(self.size_input):
                        
            for f in range(self.size_layer):
                dw = 1
                for o in range(self.size_output):
                    dw *= (self.cost[o] * self.layerow[f][o])
               
                dw *= self.dleaky_relu(self.layer1[f]) * self.input[i]
                                               
                dw += 0.01 * self.layer1w[i][f]
                adam = Adam(learning_rate = self.LR)
                self.layer1w[i][f] = adam.update(self.layer1w[i][f], dw)

        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer):
                db+= self.LR * self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer1[f])
        self.layer1b -= db #/ (self.size_output* self.size_layer)
    
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

    def backwardl2(self, batch_sz):
        
        reg = 0.01 * (np.sum(np.square(self.layer1w))+ np.sum(np.square(self.layer2w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        self.cost = self.cost / batch_sz + reg
       
        
        for o in range(self.size_output):
            for i in range(self.size_layer2):
                dw = (self.cost[o] * self.layer2[i]) + 0.01* self.layerow[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow[i][o], dw)
                
        db = 0
        for o in range(self.size_output):
            db += self.cost[o]
        self.layerob -= self.LR * db / self.size_output

        
        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                dw = 0
                for o in range(self.size_output):
                    dw += (self.cost[o] * self.layerow[f][o]) 
                dw *= self.dleaky_relu(self.layer2[f]) * self.layer1[i]
                dw += 0.01 * self.layer2w[i][f]
                adam = Adam(learning_rate = self.LR)
                self.layer2w[i][f] = adam.update(self.layer2w[i][f], dw)
                  

        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                db+=  self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f])
        self.layer2b -= self.LR * db / (self.size_output * self.size_layer2)


        for i in range(self.size_layer):
            for j in range(self.size_input):
                dw = 0
                for o in range(self.size_output):
                    for f in range(self.size_layer2):
                        dw += self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f]) * self.layer2w[i][f] 
                              
                dw *= self.dleaky_relu(self.layer1[i]) * self.input[j]
                dw += 0.01 * self.layer1w[j][i]
                adam = Adam(learning_rate = self.LR)
                self.layer1w[j][i] = adam.update(self.layer1w[j][i], dw)
        

        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                 for i in range(self.size_layer):
                    db+= (self.cost[o] * self.layerow[f][o] * self.dleaky_relu(self.layer2[f]) * self.layer2w[i][f] \
                      * self.dleaky_relu(self.layer1[i]))

        self.layer1b -= self.LR * db / (self.size_output * self.size_layer * self.size_layer2)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])
        
    def backward_youtube2(self, batch_sz):

        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layer2w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        error = (self.cost / batch_sz) + reg
        self.layerob -= self.LR * error
        dw = np.outer(error, self.layer2).T + 0.01 * self.layerow
        adam = Adam(learning_rate = self.LR)
        self.layerow = adam.update(self.layerow, dw)

        error = np.dot(self.layerow, error) * self.dleaky_relu(self.layer2)
        self.layer2b -= self.LR * error
        dw2 = np.outer(error, self.layer1).T+ 0.01 * self.layer2w
        adam = Adam(learning_rate = self.LR)
        self.layer2w = adam.update(self.layer2w, dw2)

        error = np.dot(self.layer2w, error) * self.dleaky_relu(self.layer1)
        self.layer1b -= self.LR * error
        dw3 = np.outer(error, self.input).T+ 0.01 * self.layer1w
        adam = Adam(learning_rate = self.LR)
        self.layer1w = adam.update(self.layer1w, dw3)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backwardl2u(self, batch_sz):
        
        reg = 0.1 * (np.sum(np.square(self.layer1w_s2))+ np.sum(np.square(self.layer2w_s2)) + np.sum(np.square(self.layerow_s2))) / (2*batch_sz)
       
        self.cost = self.cost / batch_sz + reg
       
        
        for o in range(self.size_output):
            for i in range(self.size_layer2):
                dw = (self.cost[o] * self.layer2_s2[i]) + 0.1* self.layerow_s2[i][o]
                adam = Adam(learning_rate = self.LR)
                self.layerow[i][o] = adam.update(self.layerow_s2[i][o], dw)
                
        db = 0
        for o in range(self.size_output):
            db += self.cost[o]
        self.layerob -= self.LR * db / self.size_output

        
        for f in range(self.size_layer2):
            for i in range(self.size_layer):
                dw = 0
                for o in range(self.size_output):
                    dw += (self.cost[o] * self.layerow_s2[f][o]) 
                dw *= self.dleaky_relu(self.layer2_s2[f]) * self.layer1_s2[i]
                dw += 0.1 * self.layer2w_s2[i][f]
                adam = Adam(learning_rate = self.LR)
                self.layer2w[i][f] = adam.update(self.layer2w_s2[i][f], dw)
                  

        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                db+=  self.cost[o] * self.layerow_s2[f][o] * self.dleaky_relu(self.layer2_s2[f])
        self.layer2b -= self.LR * db / (self.size_output * self.size_layer2)


        for i in range(self.size_layer):
            for j in range(self.size_input):
                dw = 0
                for o in range(self.size_output):
                    for f in range(self.size_layer2):
                        dw += self.cost[o] * self.layerow_s2[f][o] * self.dleaky_relu(self.layer2_s2[f]) * self.layer2w_s2[i][f] 
                              
                dw *= self.dleaky_relu(self.layer1_s2[i]) * self.input[j]
                dw += 0.1 * self.layer1w_s2[j][i]
                adam = Adam(learning_rate = self.LR)
                self.layer1w[j][i] = adam.update(self.layer1w_s2[j][i], dw)
        

        db = 0
        for o in range(self.size_output):
             for f in range(self.size_layer2):
                 for i in range(self.size_layer):
                    db+= (self.cost[o] * self.layerow_s2[f][o] * self.dleaky_relu(self.layer2_s2[f]) * self.layer2w_s2[i][f] \
                      * self.dleaky_relu(self.layer1_s2[i]))

        self.layer1b -= self.LR * db / (self.size_output * self.size_layer * self.size_layer2)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])
    
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
        if self.tp + self.fp == 0:prec = 0
        else: prec = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0:reca = 0
        else: reca = self.tp / (self.tp + self.fn)
        self.f1score = 0
        if prec+reca != 0: self.f1score = 2 * (prec * reca) / (prec + reca)
        return self.f1score

    def init_F1(self):
        self.tp = self.fn = self.fp = 0

    def calc_decay_rate(self, dr, ep):
        self.LR = 1 / ( 1 + dr * ep) * self.LR

"""nn = ConvNeuralNet(2, 5, 1, 0.00001)
for i in range(100):
    for j in range(100):
        nn.build(np.array([i, j]))
        nn.forward()
        nn.backward(np.array([i+j]))
nn.predict(np.array([200, 75]))"""

nn = ConvNeuralNet(256, 4, 4, 0.001)
#nn.add_layer2(4)

NB_IMAGE = 192
start_time = time.time()


#nn.load_file("nnmodel/61.98.txt")
#nn.load_file2("nnmodel/50.0_256x32x16x3x0.001_ly2.txt")

TEST, TEST_out, PATH, TEST2, TEST_out2, PATH2, TEST3, TEST_out3, PATH3 = create_test_tab_one_filter_rand_batch_train3(480)
## TEST, TEST_out = create_test_tab_filter()
TESTV, TESTV_out, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3 = create_test_tab_one_filter_rand_batch_train3(64)

I = 0
I2 = 0
I3 = 0
W = 0
WMAX = 9
stop = False
change_in = False
NORM = 0.1
valid = 0
valid_tot = np.zeros(200)
score = 0
pct = 0
epochs = 0

lotsz = 18
alt = 1
ilot = 0

target =np.array([1.0, 0.0, 0.0, 0.0])
target2 =np.array([0.0, 1.0, 0.0, 0.0])
target3 =np.array([0.0, 0.0, 1.0, 0.0])
pass_aug_data = False
IG = 0
IG2 = 0
VI = 0
lr = [0.001, 0.001, 0.001, 0.001]
lrs = [0.001, 0.001, 0.001, 0.001]
last_pct = 0
iter = 0
best_pct = 0
best_f1=  0
start_round = time.time()
indalt = 0
talt = [1, 2, 3]
random.shuffle(talt)
dropout = True
while not stop:

    #print("image: " + str(I) + " " + str(I2)+ " " + str(I3))
    

    if talt[indalt] == 1:
        #nn.set_lr(lr[0])
        nn.build(np.array(TEST[I]))
        #nn.input += np.random.rand() * 0.1
        #nn.input += np.random.randn(nn.size_input) * 2.0
        #nn.optimInput()
        
        #nn.BN(0.1)
        #nn.augmentation_data(2, 1)

    elif talt[indalt] == 2:
        #nn.set_lr(lr[1])
        nn.build(np.array(TEST2[I2]))
        #nn.input += np.random.rand() * 0.1
        #nn.input += np.random.randn(nn.size_input) * 2.0
        #nn.optimInput()
      
        #nn.augmentation_data2(2, 2)
    elif talt[indalt] == 3:
        #nn.set_lr(lr[2])
        nn.build(np.array(TEST3[I3]))
        #nn.input += np.random.randn(nn.size_input) * 2.0
        #nn.input += np.random.rand() * 0.1
        #nn.optimInput()
      
        #nn.augmentation_data2(2, 2)

    nn.BN(1.0)
    

    if talt[indalt] == 1:
        nn.forward_drop(target,2, dropout)
    elif talt[indalt] == 2:
        nn.forward_drop(target2, 2, dropout)
    else:
        nn.forward_drop(target3,2, dropout)

    dropout = False
    #print(nn.layero)
    #print(nn.layer1b)
    #print(nn.layerob)
     
    if talt[indalt] == 1:
        I += 1
    elif talt[indalt] == 2:
        I2+=1
    else:
        I3+=1

    indalt += 1
    if indalt == 3:
        indalt = 0
        random.shuffle(talt)

    """alt+=1
    if alt == 4:
        alt = 1"""

    ilot += 1
    if ilot == lotsz:
        #nn.save_weight_bias22()
        nn.backward_youtube(lotsz)
        ilot = 0
        dropout = True

        """TESTV = []
        TESTV_out = []
        PATHV = []
        TESTV2 = []
        TESTV_out2 = []
        PATHV2 = []
        TESTV3 = []
        TESTV_out3 = []
        PATHV3 = []
        ti = VI
        while ti < VI+lotsz*2:
            TESTV.append(TEST[ti])
            TESTV_out.append(TEST_out[ti])
            PATHV.append(PATH[ti])
            TESTV2.append(TEST2[ti])
            TESTV_out2.append(TEST_out2[ti])
            PATHV2.append(PATH2[ti])
            TESTV3.append(TEST3[ti])
            TESTV_out3.append(TEST_out3[ti])
            PATHV3.append(PATH3[ti])
            ti +=1
           
        VI+= lotsz*2
        if VI >= 448:
            VI = 0"""

        res=[0,0,0,0]
        valid = np.zeros(NB_IMAGE)
        score, pct = ValidationNN(nn, TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3, res)
        print(VI)
        for i in range(4):
            print(res[i])
            pc = res[i] / NB_IMAGE
            reg = (1 - (pc/(1.0/3.0)))
            if pc == 0:reg = 1
            #lr[i] = 0.001 * reg + 0.0000001
            #print(str(pc) + " " + str(lr[i]) + " " + str(pct))   
        if pct > last_pct and pct > 34:
            nn.save_weight_bias()
            for i in range(3):
                lrs[i]= lr[i]
            last_pct = pct
            best_pct = pct
            print("save")
        """if pct < last_pct:
            nn.load_weight_bias()
            for i in range(3):
                lr[i]= 0.001
            print("load")"""


        best_f1 = max(best_f1, nn.f1score)
        duration_round = time.time() - start_round 
        print("training : " + str((nn.loss / lotsz)*100.0))    
        print("duree : " + str(duration_round) + " s")
        print("best : " + str(best_pct) + "%")
        print("f1 score : " + str(best_f1))
        print("epoch " + str(epochs+1))
        print("LR : " + str(nn.LR))
        #print("bias " + str(nn.layer1b))
        start_round = time.time()
        nn.loss = 0
    


    if I == len(TEST) and I2 == len(TEST2) and I3 == len(TEST3):
        I  = 0
        I2 = 0
        I3 = 0
        epochs +=1
        #nn.calc_decay_rate(1, epochs)
        

    if epochs == 10:
        print("epoch: " + str(W)) 
             
        stop = True
    iter +=1

        
stop = False
I = 0
I2 = 0
ilot = 0
lotaug = 50
epochs = 0
while stop:
    print("image aug: " + str(I) + " " + str(I2))
    
    
    if alt:
        nn.build(np.array(nn.aug_dimg[I]))
     
    else:
        nn.build(np.array(nn.aug_dimg[I2]))
      
  

    nn.forward_drop(10)

    if alt:
        I += 1
    else:
        I2+=1


    ilot += 1
    if ilot == lotaug:
        if alt:
            nn.backward(target)
        else:
            nn.backward(target2)
        ilot = 0
        alt = not alt


    if I == len(nn.aug_dimg) and I2 == len(nn.aug_dimg2):
        I  = 0
        I2 = 0
        epochs +=1

    if epochs == 1:
        print("epoch: " + str(W)) 
             
        stop = True

nn.save_file("nnmodel/" + str(best_pct) + ".txt")
nn.load_weight_bias()
res = [0,0,0]
valid = np.zeros(NB_IMAGE)
score, pct = ValidationNN(nn, TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3, res)
for v in range(NB_IMAGE):
    if valid[v]:
        valid_tot[v] = 1

for i in range(3):
    print(res[i])
    pc = res[i] / NB_IMAGE
    print(str(pc) + " " + str(lr[i]))   

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
    if ilot == 32:
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