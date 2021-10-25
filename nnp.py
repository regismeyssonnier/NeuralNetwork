import tensorflow.keras.preprocessing.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
import random

size_input = 256
inputn = np.zeros(size_input)

#print(inputn)

nb_hidden = 4
size_hidden = 64
size_hidden2 = 32
size_hidden3 = 16#30#100#44
size_hiddenf = 8
max_size_hidden = size_hidden

hidden = []
hidden.append(np.zeros(size_hidden))
hidden.append(np.zeros(size_hidden2))
hidden.append(np.zeros(size_hidden3))
hidden.append(np.zeros(size_hiddenf))

#print(hidden)

hiddenw = []
hiddenw.append(np.random.random((size_hidden, size_input))-0.5)
hiddenw.append(np.random.random((size_hidden2, size_hidden))-0.5)
hiddenw.append(np.random.random((size_hidden3, size_hidden2))-0.5)
hiddenw.append(np.random.random((size_hiddenf, size_hidden3))-0.5)
#print(hiddenw)


hiddenb = []
hiddenb.append(np.random.random(size_hidden)*0.02-0.01)
hiddenb.append(np.random.random(size_hidden2)*0.02-0.01)
hiddenb.append(np.random.random(size_hidden3)*0.02-0.01)
hiddenb.append(np.random.random(size_hiddenf)*0.02-0.01)
#print(hiddenb)

sz_hidden = [size_hidden, size_hidden2]#no 
size_hidden_weight = [size_input, size_hidden]#no
size_hidden_bias = [size_hidden, size_hidden2, size_hidden3, size_hiddenf]#use

size_output = 2
size_output_weight = size_hidden2
size_output_bias = size_output

output = []
output.append(np.zeros(size_output))
outputw = []
outputw.append(np.random.random((size_output, size_hiddenf))-0.5)

outputb = []
outputb.append(np.random.random(size_output)*0.02-0.01)
#print(outputb)

LR = 0.0009

TEST = []
TEST_out = []
TESTV = []
TESTV_out = []
PATH = []
PATHV = []


def test2inputn(test, inp):
    #inp = np.zeros(len(test))
    I = 0
    for t in test:
        inp[I] = t
        I += 1
    #return inp

def display_test(test, out):

	print("output :  " + str(out))
	for i in range(16):
		d = ''
		for j in range(16):
			d += str(int(test[i*16+j])) + " "
		print(d)

#***********************************************************************************

def sigmoid(x):
	#try:
	#print("x=" + str(x))	
	#print("x:" + str(x))
	return 1 / (1 + math.exp(-x))
	"""except OverflowError:
		print("x:" + str(x))
		#print("exp:" + str(math.exp(-x)))
		return 1 / (1 + math.exp(0))
	"""
		
def softmax(x):
	if x <= 0:
		return 0
	else:
		return 1

def RELU(x):
	if x <= 0:
		return 0
	else:
		return x

def tanh(x):
	return math.tanh(x)

def derive(x):
	return x * (1 - x)

#***********************************************************************************
def calc_output_RELU2(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    nbmax = 0
    if(size_out < 4):
        nbmax = 1.0
    else:
        nbmax = size_out * 0.25
    nb = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        r = random.random() * 10
        
        if (r < 2.5) and (nb < nbmax):
            out[I] = 0
            nb += 1
        else:
            out[I]=  RELU( ri + hb[o] ) 
        I += 1
    #return out


def calc_output_RELUF(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I] = ri + hb[o]
        I += 1
    #return out


def hidden_backp41(alm1, cost, output, outputw, outputb, size_c, size_out, size_in):
        
        for c in range(size_c):
            #for o in range(size_out):
            d = derive(output[c])
            db = d * 2 * cost[c]
            outputb[c] -=  LR * db #+ random.random()
                            
            for i in range(size_in):
                
                
                dw = alm1[i] * d * 2 * cost[c]
                
                outputw[c,i] -= LR* dw


def hidden_backp4(alm1, cost, output, outputw, outputb, size_c, size_out, size_in):
        
        for c in range(size_c):
            for o in range(size_out):
                d = derive(output[o])
                db = d * 2 * cost[c]
                outputb[o] -=  LR * db #+ random.random()
                
                nb = 0
                nbmax = 0
                if(size_in < 4):
                    nbmax = 1.0
                else:
                    nbmax = size_in * 0.25
                for i in range(size_in):
                    r = random.random()
                    if (r <= 0.25) and (nb < nbmax):
                        nb += 1
                        alm1[i] = 0
                    
                    dw = alm1[i] * d * 2 * cost[c]
                    
                    outputw[o,i] -= LR* dw
                    
                    
                    
def BN(inp, size_in, scale, Y):

	s = 0
	for i in range(size_in):
		s += inp[i]

	s /= size_in

	sx = 0
	d = 0
	for i in range(size_in):
		d += (inp[i] - s)*(inp[i] - s)
		
	d /= size_in
	
	#Y = np.zeros(size_in)
	I = 0
	for i in range(size_in):
		Y[I] =  ((inp[i] - s) / np.sqrt(d + 0.000001)) * scale 
		I+=1
		
	#return Y
	


def Dropout_input(inputn):
    
    nb = 0
    nbmax = 0
    if(size_input < 4):
        nbmax = 1.0
    else:
        nbmax = size_input * 0.25
    
    #print(len(inputn))
    for i in range(size_input):
        r = random.random()
        if (r <= 0.25) and (nb < nbmax):
            nb += 1
            inputn[i] = 0


def Afficher_images(path_dir, valid, nb, score, pct):
    
    I = 1
    #ListeFichiers = os.listdir(path_dir)
    plt.figure(figsize=(15.5, 7.6), dpi=100)
   
    
    plt.subplot(2,1,1)
    Img_Pil = img.load_img(path_dir[0], target_size=(200, 200))
    plt.imshow(Img_Pil)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(score) + "/" + str(nb) + " soit " + str(pct) + "%" , pad=1, size=20, color="Blue")
    
    for NoImg in range(nb):
               
        Img_Pil = img.load_img(path_dir[NoImg], target_size=(200, 200))
        #Img_Array = img.img_to_array(Img_Pil)/255
        #Img_List = np.expand_dims(Img_Array, axis=0)
        
        plt.subplot(4,10,NoImg+1)
        plt.imshow(Img_Pil)
        plt.xticks([])
        plt.yticks([])
        
        if valid[NoImg]:
            plt.title('Bien ' + str(I) + '/' + str(nb), pad=1, size=10, color="Green")
            #plt.title(str(I), pad=1, size=10, color="Blue")
            I += 1
        else:
            plt.title('Mal classe', pad=1, size=10, color="Red")
                    
    plt.show()
    

def Validation(TESTV, TESTV_out, NORM, valid):
    
    score = 0.0
    score2 = 0.0
    score3 = 0
    score4 = 0
    nbcat = 20.0
    nbdog = 20.0

    W = 0
    WMAX = 1
    stop = False
    change_in = True
    R = [0, 0]
    reso = 0
    reso2 = 0.0
    reso3 = 0
    reso4 = 0
    I = 0
                
    while not stop:

        if change_in:
            #print(len(TESTV))
            test2inputn(TESTV[I], inputn)
            #display_test(inputn, TESTV_out[I])
            BN(inputn, size_input, NORM, inputn)
            #display_test(inputn, TESTV_out[I])
            change_in = False
        
        calc_output_RELU2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        calc_output_RELU2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        calc_output_RELU2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        calc_output_RELU2(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        calc_output_RELUF(hidden[3], outputw[0], outputb[0], size_hiddenf, size_output_bias, output[0])

        #print(str(I) + " : " + str(output[0]))
        
        
        maxi = -1
        resom = -1
        for o in range(size_output):
            if output[0][o] > maxi:
                resom = o+1
                maxi = output[0][o]

        name = ""
        if (TESTV_out[I]) == 1:
            name = "cat"
        elif(TESTV_out[I]) == 2:
            name="dog"
            
        #print(str(I) + " : " + name)
        
        if resom == 1:
            #print("cat")
            if TESTV_out[I] == 1:
                score += 1
                valid[I] = True
            else:
                valid[I] = False
        elif resom == 2:
            #print("dog")
            if TESTV_out[I] == 2:
                score += 1
                valid[I] = True
            else:
                valid[I] = False

        I += 1
        change_in = True

        cnt = 0	
        E = 0		

        if I == len(TESTV):
            I = 0
            W+=1

            if W == WMAX:
                stop = True
                

    pct = float(score/(nbcat+nbdog))*100.0
    print("Algo Score:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
    
    print("---------------------------------------------------------------------------------")

    return score, pct
















