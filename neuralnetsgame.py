import time
import random
import numpy as np
from nnp import *
from im import *
from save import *
import time
import random
import matplotlib.pyplot as plt




class BatchNorm:
    def __init__(self, input_size, lr, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((input_size))
        self.beta = np.zeros((input_size))
        self.running_mean = np.zeros((input_size))
        self.running_var = np.zeros((input_size))
        self.lr = lr
    
    def forward(self, x):
        # Calcule la moyenne et la variance de la batch
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # Normalise la batch
        x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

        # Applique l'échelle et le décalage
        out = self.gamma * x_norm + self.beta

        # Met à jour la moyenne et la variance mobiles pour la phase de test
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        return out

    def forward_pred(self, x):
        
        # Normalise la batch
        x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Applique l'échelle et le décalage
        out = self.gamma * x_norm + self.beta

        return out

    def backward(self, x, grad_out):
        # Calculer les dérivées des paramètres
        dgamma = np.sum(grad_out * x, axis=0)
        dbeta = np.sum(grad_out, axis=0)

        # Calculer la dérivée de la batch normalisée
        dx_norm = grad_out * self.gamma

        # Calculer la dérivée de la variance et de la moyenne de la batch
        N = x.shape[0]
        var = 1.0 / (N * (self.running_var + self.eps))
        dvar = np.sum(dx_norm * (x - self.running_mean), axis=0, keepdims=True) * -0.5 * var**1.5
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.running_var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - self.running_mean), axis=0, keepdims=True)

        # Calculer la dérivée de l'entrée
        dx = dx_norm / np.sqrt(self.running_var + self.eps) + dvar * 2 * (x - self.running_mean) / N + dmean / N

        # Mettre à jour les paramètres
        self.gamma -= self.lr * dgamma
        self.beta -= self.lr * dbeta

        return dx



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

    def __init__(self, sz_inp, size, outsz, lr, _lambda):
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
        self._lambda = _lambda

        """
        self.layer1w = (np.random.random((self.size_input, self.size_layer))*40.0+1)/100.0
        self.layerow = (np.random.random((self.size_layer, self.size_output))*40.0+1)/100.0
        self.layer1b = (np.random.random(size)*40.0+1)/100.0
        self.layerob = (np.random.random(outsz)*40.0+1)/100.0"""

        self.layer1w = np.random.randn(self.size_input, self.size_layer) / np.sqrt(self.size_input)
        self.layerow = np.random.randn(self.size_layer, self.size_output) / np.sqrt(self.size_layer)
        self.layer1b = np.random.randn(self.size_layer)
        self.layerob = np.random.randn(self.size_output)
        self.batchnorml1 = BatchNorm(self.size_layer, lr)
        self.batchnorml1_s = BatchNorm(self.size_layer, lr)
        self.adamow = Adam(learning_rate=lr)
        self.adamob = Adam(learning_rate=lr)
        self.adam1w = Adam(learning_rate=lr)
        self.adam1b = Adam(learning_rate=lr)

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
        self.layer2b = np.random.randn(self.size_layer2)
        self.batchnorml2 = BatchNorm(self.size_layer2, self.LR)
        self.batchnorml2_s = BatchNorm(self.size_layer2, self.LR)
        self.adam2w = Adam(learning_rate=self.LR)
        self.adam2b = Adam(learning_rate=self.LR)

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

        for i in range(self.size_layer):
            self.batchnorml1_s.running_mean[i] = self.batchnorml1.running_mean[i]

        for i in range(self.size_layer):
            self.batchnorml1_s.running_var[i] = self.batchnorml1.running_var[i]


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

        for i in range(self.size_layer):
            self.batchnorml1_s.running_mean[i] = self.batchnorml1.running_mean[i]

        for i in range(self.size_layer):
            self.batchnorml1_s.running_var[i] = self.batchnorml1.running_var[i]

        for i in range(self.size_layer2):
            self.batchnorml2_s.running_mean[i] = self.batchnorml2.running_mean[i]

        for i in range(self.size_layer2):
            self.batchnorml2_s.running_var[i] = self.batchnorml2.running_var[i]

        

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

        for l in range(self.size_layer):
            f.write(str(self.batchnorml1_s.running_mean[l])+ "\n")

        for l in range(self.size_layer):
            f.write(str(self.batchnorml1_s.running_var[l])+ "\n")

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

        for l in range(self.size_layer):
            f.write(str(self.batchnorml1_s.running_mean[l])+ "\n")

        for l in range(self.size_layer):
            f.write(str(self.batchnorml1_s.running_var[l])+ "\n")

        for l in range(self.size_layer2):
            f.write(str(self.batchnorml2_s.running_mean[l])+ "\n")

        for l in range(self.size_layer2):
            f.write(str(self.batchnorml2_s.running_var[l])+ "\n")

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

        for l in range(self.size_layer):
            self.batchnorml1_s.running_mean[l] = float(f.readline())

        for l in range(self.size_layer):
            self.batchnorml1_s.running_var[l] = float(f.readline())

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

        for l in range(self.size_layer):
            self.batchnorml1_s.running_mean[l] = float(f.readline())

        for l in range(self.size_layer):
            self.batchnorml1_s.running_var[l] = float(f.readline())

        for l in range(self.size_layer2):
            self.batchnorml2_s.running_mean[l] = float(f.readline())

        for l in range(self.size_layer2):
            self.batchnorml2_s.running_var[l] = float(f.readline())

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

        for i in range(self.size_layer):
            self.batchnorml1.running_mean[i] = self.batchnorml1_s.running_mean[i]

        for i in range(self.size_layer):
            self.batchnorml1.running_var[i] = self.batchnorml1_s.running_var[i]

        

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

        for i in range(self.size_layer):
            self.batchnorml1.running_mean[i] = self.batchnorml1_s.running_mean[i]

        for i in range(self.size_layer):
            self.batchnorml1.running_var[i] = self.batchnorml1_s.running_var[i]

        for i in range(self.size_layer2):
            self.batchnorml2.running_mean[i] = self.batchnorml2_s.running_mean[i]

        for i in range(self.size_layer2):
            self.batchnorml2.running_var[i] = self.batchnorml2_s.running_var[i]
        

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

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x)**2


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

    def predict_softmax_norm(self):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)
        forward = np.dot(self.input, self.layer1w)+ self.layer1b
        forwardp = self.batchnorml1.forward_pred(forward)
        self.layer1 = self.tanh(forwardp)
        
        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer1, self.layerow + self.layerob))

    def predict_softmax_norm2(self):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        forward = np.dot(self.input, self.layer1w)+ self.layer1b
        forwardp = self.batchnorml1.forward_pred(forward)
        self.layer1 = self.tanh(forwardp)

        forward = np.dot(self.layer1, self.layer2w)+ self.layer2b
        forwardp = self.batchnorml2.forward(forward)
        self.layer2 = self.tanh(forwardp)
             
        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer2, self.layerow + self.layerob))

    def forward_drop(self, target, nb, ndrop):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée

        forward = np.dot(self.input, self.layer1w)+ self.layer1b
        forwardp = self.batchnorml1.forward(forward)
        hidden_output = self.tanh(forwardp)
        if ndrop:
            drop = [0]*nb
            drop += [1]*(self.size_layer-nb)
            #self.dropout_mask = np.random.binomial(1, 0.7, size=hidden_output.shape)
            random.shuffle(drop)
            self.dropout_mask = np.array(drop)
        #print(dropout_mask)
     
        self.layer1 = hidden_output * self.dropout_mask 
        
        # Calculer la sortie finale
        self.layero = self.softmax(np.dot(self.layer1, self.layerow) + self.layerob)
   
        self.cost += (self.layero - target)
        #self.cost += (self.layero[np.newaxis, :] - target[np.newaxis, :])

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

    def forward_drop2(self, target, nb, nb2, ndrop):
        
        #self.layer1 = self.leaky_relu(np.dot(self.input, self.layer1w) + self.layer1b)
        #self.layero = self.leaky_relu(np.dot(self.layer1, self.layerow) + self.layerob)

        # Ajouter une étape de dropout après la couche cachée
        forward = np.dot(self.input, self.layer1w)+ self.layer1b
        forwardp = self.batchnorml1.forward(forward)
        hidden_output = self.tanh(forwardp)
        if ndrop:
            drop = [0]*nb
            drop += [1]*(self.size_layer-nb)
            #self.dropout_mask = np.random.binomial(1, 0.7, size=hidden_output.shape)
            random.shuffle(drop)
            self.dropout_mask = np.array(drop)
        #print(dropout_mask)
        self.layer1 = hidden_output * self.dropout_mask

        forward = np.dot(self.layer1, self.layer2w)+ self.layer2b
        forwardp = self.batchnorml2.forward(forward)
        hidden_output2 = self.tanh(forwardp)
       
        if ndrop:
            drop = [0]*nb2
            drop += [1]*(self.size_layer2-nb2)
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
        print(dw.shape)
        adam = Adam(learning_rate = self.LR)
        self.layerow = adam.update(self.layerow, dw)

        error = np.dot(self.layerow, error) * self.dleaky_relu(self.layer1)
        self.layer1b -= self.LR * error
        dw2 = np.outer(error, self.input).T+ 0.01 * self.layer1w
        adam = Adam(learning_rate = self.LR)
        self.layer1w = adam.update(self.layer1w, dw2)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backward_youtube_rm(self, batch_sz):

        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
       
        error = (self.cost / batch_sz) + reg
        self.layerob -= self.LR * error
        dw = np.outer(self.layer1, error) + 0.01 * self.layerow
        adam = Adam(learning_rate = self.LR)
        self.layerow = adam.update(self.layerow, dw)

        error = np.dot(self.layerow, error) * self.dleaky_relu(self.layer1)
        self.layer1b -= self.LR * error
        dw2 = np.outer(self.input, error)+ 0.01 * self.layer1w
        adam = Adam(learning_rate = self.LR)
        self.layer1w = adam.update(self.layer1w, dw2)

        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backward_youtube_rm_norm(self, batch_sz):
        # Calcul de la régularisation L2
        reg = self._lambda * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)

        # Calcul de l'erreur de sortie
        error = (self.cost / batch_sz) + reg
           

        # Calcul de la dérivée de la sortie
        grad_out = error.reshape(1, -1)

        # Rétropropagation à travers la couche de sortie
        dx = np.dot(grad_out, self.layerow.T)
        db = np.sum(grad_out, axis=0)
        # Mise à jour des paramètres de la couche de sortie
        dw = np.outer(self.layer1, grad_out) + self._lambda * self.layerow**2
        self.layerow = self.adamow.update(self.layerow, dw)
        self.layerob = self.adamob.update(self.layerob, db)

        # Rétropropagation à travers la couche cachée
        dx = dx * self.dtanh(self.layer1)

        # Normalisation de la batch avant la rétropropagation
        #bn = BatchNorm(self.size_layer)
        dx_norm = self.batchnorml1.backward(self.layer1, dx)

        # Rétropropagation à travers la couche cachée avec batch normalization
        dx = np.dot(dx_norm, self.layer1w.T)
        #dw = np.dot(self.input.T, dx_norm)
        db = np.sum(dx_norm, axis=0)

        # Mise à jour des paramètres de la couche cachée
        dw = np.outer(self.input, dx_norm) + self._lambda * self.layer1w**2
        self.layer1w = self.adam1w.update(self.layer1w, dw)
        self.layer1b = self.adam1b.update(self.layer1b, db)

        # Réinitialisation du coût
        self.cost = np.array([0.0, 0.0, 0.0, 0.0])

    def backward_youtube_rm_norm2(self, batch_sz):
        # Calcul de la régularisation L2
        reg = self._lambda * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layer2w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)

        # Calcul de l'erreur de sortie
        error = (self.cost / batch_sz) + reg
           

        # Calcul de la dérivée de la sortie
        grad_out = error.reshape(1, -1)

        # Rétropropagation à travers la couche de sortie
        dx = np.dot(grad_out, self.layerow.T)
        db = np.sum(grad_out, axis=0)
        # Mise à jour des paramètres de la couche de sortie
        dw = np.outer(self.layer2, grad_out) + self._lambda * self.layerow**2
        self.layerow = self.adamow.update(self.layerow, dw)
        self.layerob = self.adamob.update(self.layerob, db)

        # Rétropropagation à travers la couche cachée
        dx = dx * self.dtanh(self.layer2)

        # Normalisation de la batch avant la rétropropagation
        dx_norm = self.batchnorml2.backward(self.layer2, dx)

        # Rétropropagation à travers la couche cachée avec batch normalization
        dx = np.dot(dx_norm, self.layer2w.T)
        db = np.sum(dx_norm, axis=0)

        # Mise à jour des paramètres de la couche cachée
        dw = np.outer(self.layer1, dx_norm) + self._lambda * self.layer2w**2
        self.layer2w = self.adam2w.update(self.layer2w, dw)
        self.layer2b = self.adam2b.update(self.layer2b, db)

        dx = dx * self.tanh(self.layer1)
        dx_norm = self.batchnorml1.backward(self.layer1, dx)

        db = np.sum(dx_norm, axis=0)
        dw = np.outer(self.input, dx_norm) + self._lambda * self.layer1w**2
        self.layer1w = self.adam1w.update(self.layer1w, dw)
        self.layer1b = self.adam1b.update(self.layer1b, db)

        # Réinitialisation du coût
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

    def backward_youtube_gpt(self, batch_sz):
        reg = 0.01 * (np.sum(np.square(self.layer1w)) + np.sum(np.square(self.layerow))) / (2*batch_sz)
        error = (self.cost / batch_sz) + reg
    
        # Mise à jour des poids de la couche de sortie
        dlayerob = error
        self.layerob -= self.LR * dlayerob
        dw_layerow = np.outer(dlayerob, self.layer1) + 0.01 * self.layerow
        adam = Adam(learning_rate=self.LR)
        self.layerow = adam.update(self.layerow, dw_layerow)
    
        # Calcul de l'erreur de la couche cachée
        dlayer1 = np.dot(self.layerow.T, dlayerob) * self.dleaky_relu(self.layer1)
    
        # Mise à jour des poids de la couche cachée
        dlayer1b = dlayer1
        dw_layer1w = np.outer(dlayer1b, self.input).T + 0.01 * self.layer1w
        adam = Adam(learning_rate=self.LR)
        self.layer1w = adam.update(self.layer1w, dw_layer1w)
    
        # Mise à jour du biais de la couche cachée et calcul de la nouvelle erreur
        self.layer1b -= self.LR * dlayer1b
        error = dlayerob
    
        # Mise à jour de la fonction de coût
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

nbinput = 4096
layer1 = 2048
layer2 = 2048

nn = ConvNeuralNet(nbinput, layer1, 4, 0.001, 0.001)
nn.add_layer2(layer2)

NB_IMAGE = 192
start_time = time.time()

history = {}
history["accuracy"] = []
history["loss"] = []
history["val_accuracy"] = []
history["val_loss"] = []
nbhist = 0

#nn.load_file("nnmodel/61.98x256x4x4x0.001.txt")
#nn.load_file("nnmodel/41.66x256x256x4x0.001xd64rm.txt")
#nn.load_file2("nnmodel/41.66666666666667x1920x256x256ly2.txt")

TEST, TEST_out, PATH, TEST2, TEST_out2, PATH2, TEST3, TEST_out3, PATH3 = create_test_tab_one_filter_rand_batch_train3(480)
## TEST, TEST_out = create_test_tab_filter()
TESTV, TESTV_out, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3 = create_test_tab_one_filter_rand_batch_train3(64)

lotsz = 64

TEST = Normalize_img_batch(TEST, lotsz)
TEST2 = Normalize_img_batch(TEST2, lotsz)
TEST3 = Normalize_img_batch(TEST3, lotsz)

TESTV = Normalize_img_batch(TESTV, lotsz)
TESTV2 = Normalize_img_batch(TESTV2, lotsz)
TESTV3 = Normalize_img_batch(TESTV3, lotsz)

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

    #nn.BN(1.0)
    

    if talt[indalt] == 1:
        nn.forward_drop2(target,1024, 1, dropout)
    elif talt[indalt] == 2:
        nn.forward_drop2(target2,1024, 1, dropout)
    else:
        nn.forward_drop2(target3,1024, 1, dropout)

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
        nn.backward_youtube_rm_norm2(lotsz)
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
            nn.save_weight_bias2()
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

        #print(nn.layer1w)
        #print(nn.layerow)

        history["accuracy"].append((nn.loss / lotsz)*100.0)
        history["loss"].append(100-((nn.loss / lotsz)*100.0))
        history["val_accuracy"].append(pct)
        history["val_loss"].append(100-pct)

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
        
        nbhist+=1


    if I == len(TEST) and I2 == len(TEST2) and I3 == len(TEST3):
        I  = 0
        I2 = 0
        I3 = 0
        epochs +=1
        #nn.calc_decay_rate(1, epochs)
        

    if epochs == 200:
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

nn.save_file2("nnmodel/" + str(best_pct) +"x" + str(nbinput) +"x" + str(layer1) +"x" + str(layer2) + "ly2.txt")
nn.load_weight_bias2()
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

acc = history['accuracy']
val_acc = history['val_accuracy']

print("nbhist : " + str(nbhist))
loss = history['loss']
val_loss = history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
epochs_range = range(nbhist)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

Afficher_images2(PATHF, valid_tot, NB_IMAGE, score, pct)
#print(nn.input)
#print(nn.layero)
#print(nn.layer1w)
#print(nn.layerow)