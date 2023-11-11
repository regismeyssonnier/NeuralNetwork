# coding: utf-8
import numpy as np
import random
from im import *

class NeuralNetwork:

    def __init__(self, dimension, LR):
        self.network  =[]
        self.network_w = []
        self.network_b = []
        self.etiquette = []
        self.LR = LR
        self.LRB = LR

        

        # Initialisation des paramètres d'Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.epoch = 0 

        i = 0
        for dim in dimension:
            self.network.append([0.0]*dim)

            if i > 0:
                self.network_b.append(np.random.uniform(low=-0.5, high=0.5, size=(dim)))

            if i < len(dimension)-1:
                self.network_w.append(np.random.uniform(low=-0.5, high=0.5, size=(dim, dimension[i+1])))


            i+=1


        self.m_w = [[[0.0] * len(k) for k in j] for j in self.network_w]
        self.v_w = [[[0.0] * len(k) for k in j] for j in self.network_w]
        self.m_b = [[0.0] * len(k) for k in self.network_b]
        self.v_b = [[0.0] * len(k) for k in self.network_b]

        self.cost = [0.0] * dimension[-1]

        #print(self.network_w)

    def sigmoid(self, x):
        x = np.clip(x, -700, 700)  # Ajustez la plage selon vos besoins
        return 1 / (1 + np.exp(-x))
        

    def derive(self, x):
        return x * (1 - x)

    def Relu(self, x):
        return np.maximum(0, x)

    def dRELU(self, x):
        return np.greater(x, 0)

    def SetEtiquette(self, et):
        self.etiquette = et;

    def SetInput(self, inp):
        self.network[0] = inp

              
    def ForwardNN(self):

        for i in range(0, len(self.network)-2):
            for j in  range(0, len(self.network[i+1])):
                h = 0.0
                for k in range(0, len(self.network[i])):
                    h += self.network[i][k] * self.network_w[i][k][j]

                h += self.network_b[i][j];
                self.network[i+1][j] = self.Relu(h);


        ind = len(self.network)-2
        for j in range(len(self.network[ind+1])):
            h = 0.0
            for k in range(len(self.network[ind])):
                h += self.network[ind][k] * self.network_w[ind][k][j]
            h += self.network_b[ind][j];
            self.network[ind + 1][j] = self.sigmoid(h);

        
        for i in range(len(self.network[-1])):
            self.cost[i] += self.network[-1][i] - self.etiquette[i]

    def PredictNN(self):

        for i in range(0, len(self.network)-2):
            for j in  range(0, len(self.network[i+1])):
                h = 0.0
                for k in range(0, len(self.network[i])):
                    h += self.network[i][k] * self.network_w[i][k][j]

                h += self.network_b[i][j];
                self.network[i+1][j] = self.Relu(h);


        ind = len(self.network)-2
        for j in range(len(self.network[ind+1])):
            h = 0.0
            for k in range(len(self.network[ind])):
                h += self.network[ind][k] * self.network_w[ind][k][j]
            h += self.network_b[ind][j];
            self.network[ind + 1][j] = self.sigmoid(h);


    def BackwardNN(self):
       
        result = [ [] for _ in range(len(self.network)-1) ]
        
        for i in range(len(self.network[-1])):
            result[-1].append(self.cost[i] * self.derive(self.network[-1][i]))
            self.network_b[-1][i] -= self.LR * result[-1][i]


        for i in range(len(result) - 2, -1, -1):
            #print(i)
            for j in range(len(self.network[i + 1])):
                result[i].append(0.0)
                for k in range(len(self.network[i + 2])):
                    result[i][j] += self.network_w[i + 1][j][k] * result[i + 1][k] * self.dRELU(self.network[i + 1][j])
                self.network_b[i][j] -= self.LR * result[i][j]

        #print(result)

        for i in range(len(self.network) - 2, -1, -1):
            for j in range(len(self.network[i])):
                for k in range(len(self.network[i + 1])):
                    self.network_w[i][j][k] -= self.LR * self.network[i][j] * result[i][k]

        self.cost = [0.0] * len(self.network[-1])

    def BackwardNNA(self):
       
        result = [ [] for _ in range(len(self.network)-1) ]
        
        for i in range(len(self.network[-1])):
            result[-1].append(self.cost[i] * self.derive(self.network[-1][i]))
            self.network_b[-1][i] -= self.LR * result[-1][i]


        for i in range(len(result) - 2, -1, -1):
            #print(i)
            for j in range(len(self.network[i + 1])):
                result[i].append(0.0)
                for k in range(len(self.network[i + 2])):
                    result[i][j] += self.network_w[i + 1][j][k] * result[i + 1][k] * self.dRELU(self.network[i + 1][j])
                self.network_b[i][j] -= self.LR * result[i][j]

        
        # Mise à jour des poids avec l'optimiseur Adam
        for i in range(len(self.network) - 2, -1, -1):
            for j in range(len(self.network[i])):
                for k in range(len(self.network[i + 1])):
                    # Calcul des gradients pour les poids
                    gradient_w = self.network[i][j] * result[i][k]

                    # Mise à jour d'Adam
                    self.m_w[i][j][k] = self.beta1 * self.m_w[i][j][k] + (1 - self.beta1) * gradient_w
                    self.v_w[i][j][k] = self.beta2 * self.v_w[i][j][k] + (1 - self.beta2) * (gradient_w ** 2)

                    m_hat_w = self.m_w[i][j][k] / (1 - self.beta1 ** (self.epoch + 1))
                    v_hat_w = self.v_w[i][j][k] / (1 - self.beta2 ** (self.epoch + 1))

                    self.network_w[i][j][k] -= self.LR * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

        # Mise à jour des biais avec l'optimiseur Adam
        for i in range(len(self.network) - 2, -1, -1):
            for j in range(len(self.network[i+1])):
                # Calcul des gradients pour les biais
                gradient_b = result[i][j]

                # Mise à jour d'Adam
                self.m_b[i][j] = self.beta1 * self.m_b[i][j] + (1 - self.beta1) * gradient_b
                self.v_b[i][j] = self.beta2 * self.v_b[i][j] + (1 - self.beta2) * (gradient_b ** 2)

                m_hat_b = self.m_b[i][j] / (1 - self.beta1 ** (self.epoch + 1))
                v_hat_b = self.v_b[i][j] / (1 - self.beta2 ** (self.epoch + 1))

                self.network_b[i][j] -= self.LR * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)



        self.cost = [0.0] * len(self.network[-1])

        self.epoch += 1

    def TrainingNN(self, inp, target, nbi, nb):
     
        indexes = list(range(nbi))

        for i in range(nb):
            random.shuffle(indexes)

            for jj in range(nbi):
                j = indexes[jj]
                self.SetInput(inp[j])
                self.SetEtiquette(target[j])
                self.ForwardNN()
                self.BackwardNN()

                if (i + 1) % 1000 == 0:
                    cost_ = [0.0] * len(self.network[-1])

                    for k in range(nbi):
                        self.SetInput(inp[k])
                        self.SetEtiquette(target[k])
                        self.ForwardNN()

                        for l in range(len(self.network[-1])):
                            cost_[l] += (self.network[-1][l] - target[k][l]) ** 2

                    for l in range(len(self.network[-1])):
                        cost_[l] = cost_[l] / len(self.network[-1])

                    for l in range(len(self.network[-1])):
                        print("output {}: {}".format(l, self.network[-1][l]))

                    for l in range(len(self.network[-1])):
                        print("error {}: {}".format(l, cost_[l]))


    def Save_NN(self, name):
        f = open("Test/" + name, "w")

        for i in range(len(self.network_w)):
            for j in range(len(self.network_w[i])):
                for k in range(len(self.network_w[i][j])):
                    f.write(str(self.network_w[i][j][k])+'\n')

        for i in range(len(self.network_b)):
            for j in range(len(self.network_b[i])):
                f.write(str(self.network_b[i][j])+'\n')

        f.close()

    def Load_NN(self, name):
        f = open("Test/" + name, "r")

        for i in range(len(self.network_w)):
            for j in range(len(self.network_w[i])):
                for k in range(len(self.network_w[i][j])):
                    self.network_w[i][j][k] = float(f.readline())


        for i in range(len(self.network_b)):
            for j in range(len(self.network_b[i])):
                self.network_b[i][j] = float(f.readline())

        f.close()

    def Adapt_LR(self, good, nb):

        self.LR = (self.LRB / nb) * ((nb+0.0001) - good)
            

    def DisplayOutputNN(self):
        print(self.network_w)
        print(self.network_b)
        print("output:")
        for i in range(len(self.network[-1])):
            print("{} {}".format(i, self.network[-1][i]))


TEST, TEST_out, PATH, TEST2, TEST_out2, PATH2 = create_test_tab_one_filter_rand_CrC(5)
TESTV, TESTV_out, PATHV, TESTV2, TESTV_out2, PATHV2 = create_test_tab_one_filter_rand_CrC(5)

nn = NeuralNetwork([256, 128, 64, 32, 16, 8, 2], 0.01)

#256, 128, 64, 32, 16, 8, 2 0.1

#nn.Load_NN("best7.txt")

index_class = 0
index_img1 = 0
index_img2 = 0

EPOCHS = 0
MAX_EPOCHS = 100

best = -1
sbest= -1
indexes = [0,0,0,0,0,1,1,1,1,1]
indx = 0
random.shuffle(indexes)
mod = 100
while(EPOCHS < MAX_EPOCHS):

    #index_class = random.randint(0, 1)
    index_class = indexes[indx]

    indx+=1
    if(indx == 10):
        indx = 0
        random.shuffle(indexes)


    if index_class == 0:
        nn.SetEtiquette([1, 0])
        nn.SetInput(TEST[index_img1])
        index_img1+=1
        if(index_img1 == 5):
            #index_class = 1
            index_img1 = 0
            index_img2 = 0

    elif index_class == 1:
        nn.SetEtiquette([0, 1])
        nn.SetInput(TEST2[index_img2])
        index_img2+=1
        if(index_img2 == 5):
            #index_class = 0
            index_img2 = 0
            index_img1 = 0

    nn.ForwardNN()
    nn.BackwardNNA()

    if(EPOCHS % mod == 0):
        mod = random.randint(5,10)
        print(EPOCHS)
        good = 0
        sgood = 0
        i1 = i2 = 0
        indexest = [0,0,0,0,0,1,1,1,1,1]
        random.shuffle(indexest)
        for i in range(len(indexest)):
            target = []
            if indexest[i] == 0:
                target = [1,0]
                nn.SetInput(TESTV[i1])
            else:
                target = [0,1]
                nn.SetInput(TESTV2[i2])
            nn.SetEtiquette(target)
            nn.PredictNN()

            ind = -1
            maxi = -1
            for l in range(len(nn.network[-1])):
                if l == indexest[i] and nn.network[-1][l] > 0.5:
                    sgood += 1

                if nn.network[-1][l] > maxi:
                    maxi = nn.network[-1][l]
                    ind = l

            if indexest[i] == 0:
                i1+=1
                if ind == 0:
                    good+=1
            else:
                i2+=1
                if ind == 1:
                    good+=1

           
            cost_ = [0.0] * len(nn.network[-1])
            for l in range(len(nn.network[-1])):
                cost_[l] += (nn.network[-1][l] - target[l]) ** 2

        if good > best:
            best = good
            nn.Save_NN("best" + str(best) + ".txt")

        if sgood > sbest:
            sbest = sgood
            nn.Save_NN("sbest" + str(sbest) + ".txt")

        #nn.Adapt_LR(good, 10)
        print("LR adapt : " + str(nn.LR))

        print("good "  + str(good))
        print("sgood "  + str(sgood))
        for l in range(len(nn.network[-1])):
            cost_[l] = cost_[l] / len(indexes)

        #for l in range(len(nn.network[-1])):
         #   print("output {}: {} a {:.0f}%".format(l, nn.network[-1][l], nn.network[-1][l] * 100))

        for l in range(len(nn.network[-1])):
            print("error {}: {}".format(l, cost_[l]))
            

    EPOCHS+=1

nn.DisplayOutputNN()

nn.Load_NN("best" + str(best) + ".txt")

index_img1 = 0
index_img2 = 0
index_class = 0
pct1 = 0
pct2 = 0

good = 0
while True:

    print(str(index_img1) + " " + str(index_img2))

    if index_class == 0:
        print(PATHV[index_img1])
        nn.SetEtiquette([1, 0])
        nn.SetInput(TESTV[index_img1])
    

    elif index_class == 1:
        print(PATHV2[index_img2])
        nn.SetEtiquette([0, 1])
        nn.SetInput(TESTV2[index_img2])
             

    nn.PredictNN()
               

    ind = 0
    maxi = -1
    for l in range(len(nn.network[-1])):
        if nn.network[-1][l] > maxi:
            maxi = nn.network[-1][l]
            ind = l


    for l in range(len(nn.network[-1])):
        print("output {}: {:.4f} a {:.1f}%".format(l, nn.network[-1][l], nn.network[-1][l] * 100))
        if l == 0:
            pct1 += nn.network[-1][l] *100
        elif l == 1:
            pct2 += nn.network[-1][l]*100

    if index_class == 0 and ind == 0:
        print("JUSTE")
        good+=1
    if index_class == 0 and ind != 0:
        print("FAUX")

    if index_class == 1 and ind == 1:
        print("JUSTE")
        good+=1

    if index_class == 1 and ind != 1:
        print("FAUX")

    if index_class == 0:
        index_img1+=1
        if(index_img1 == 5):
            index_class = 1
            index_img2 = 0

    elif index_class == 1:
        index_img2+=1
        if(index_img2 == 5):
            index_class = 1
            index_img2 = 0
            break
    

print("PCT : " +  str(good / 10.0 * 100.0) + "%")
print("Soit Croix :{:.1f}% Cercle {:.1f}%".format(pct1/10, pct2/10))





"""
input_data = [
    [0, 0], 
    [0, 1],
    [1, 0], 
    [1, 1], 
] 
      
output_data = [
    [1, 0],
    [0, 1],
    [0, 1], 
    [1, 0],
]

nn.TrainingNN(input_data,output_data, 4, 10000)

nn.SetEtiquette([0,0]);
nn.SetInput([ 0, 1 ]);
nn.ForwardNN();
nn.DisplayOutputNN();
nn.SetInput([ 0, 0 ]); 
nn.ForwardNN();
nn.DisplayOutputNN();
nn.SetInput([ 1, 1 ]);
nn.ForwardNN();
nn.DisplayOutputNN();
nn.SetInput([ 1, 0 ]);
nn.ForwardNN();
nn.DisplayOutputNN()

"""