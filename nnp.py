import tensorflow.keras.preprocessing.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math

size_input = 12288
inputn = np.zeros(size_input)

#print(inputn)

nb_hidden = 1
size_hidden = 128
size_hidden2 = 4
size_hidden3 = 4#30#100#44
size_hiddenf = 4
max_size_hidden = size_hidden

hidden = []
hidden.append(np.zeros(size_hidden))
hidden.append(np.zeros(size_hidden2))
hidden.append(np.zeros(size_hidden3))
hidden.append(np.zeros(size_hiddenf))

hidden2 = []
hidden2.append(np.zeros(size_hidden))
hidden2.append(np.zeros(size_hidden2))
hidden2.append(np.zeros(size_hidden3))
hidden2.append(np.zeros(size_hiddenf))

#print(hidden)

hiddenw = []

"""
hiddenw.append((np.random.random((size_hidden, size_input))*40.0+1)/100.0)
hiddenw.append((np.random.random((size_hidden2, size_hidden))*40.0+1)/100.0)
hiddenw.append((np.random.random((size_hidden3, size_hidden2))*40.0+1)/100.0)
hiddenw.append((np.random.random((size_hiddenf, size_hidden3))*40.0+1)/100.0)
"""

#hiddenw.append(np.random.randn(size_hidden, size_input) / np.sqrt(size_hidden))
#hiddenw.append(np.random.randn(size_hidden, size_input) * np.sqrt(1/size_hidden))
hiddenw.append(np.random.randn(size_hidden, size_input) / np.sqrt(size_input))


#print(hiddenw)
hiddenw2 = []
hiddenw2.append(np.zeros((size_hidden, size_input)))
hiddenw2.append(np.zeros((size_hidden2, size_hidden)))
hiddenw2.append(np.zeros((size_hidden3, size_hidden2)))
hiddenw2.append(np.zeros((size_hiddenf, size_hidden3)))

hiddenb = []
hiddenb.append(np.zeros(size_hidden))
hiddenb.append(np.zeros(size_hidden2))
hiddenb.append(np.zeros(size_hidden3))
hiddenb.append(np.zeros(size_hiddenf))

"""
hiddenb.append((np.random.random(size_hidden)*40.0+1)/100.0)
hiddenb.append((np.random.random(size_hidden2)*40.0+1)/100.0)
hiddenb.append((np.random.random(size_hidden3)*40.0+1)/100.0)
hiddenb.append((np.random.random(size_hiddenf)*40.0+1)/100.0)"""
#print(hiddenb)

hiddenb2 = []
hiddenb2.append(np.zeros(size_hidden))
hiddenb2.append(np.zeros(size_hidden2))
hiddenb2.append(np.zeros(size_hidden3))
hiddenb2.append(np.zeros(size_hiddenf))

sz_hidden = [size_hidden, size_hidden2]#no 
size_hidden_weight = [size_input, size_hidden]#no
size_hidden_bias = [size_hidden, size_hidden2, size_hidden3, size_hiddenf]#use

size_output = 3
size_output_weight = size_hidden2
size_output_bias = size_output

output = []
output.append(np.zeros(size_output))
output2 = []
output2.append(np.zeros(size_output))

outputw = []
#outputw.append(np.random.random((size_output, size_hiddenf))*40.0/100.0)
#outputw.append((np.random.random((size_output, size_hidden))*40.0+1)/100.0)
outputw.append(np.random.randn(size_output, size_hidden) / np.sqrt(size_hidden))

outputw2 = []
#outputw2.append(np.zeros((size_output, size_hiddenf)))
outputw2.append(np.zeros((size_output, size_hidden)))

outputb = []
#outputb.append((np.random.random(size_output)*40.0+1)/100.0)
outputb.append(np.zeros(size_output))
#print(outputb)
outputb2 = []
outputb2.append(np.zeros(size_output))

LR = 0.00001
LRB =0.0000000000001

TEST = []
TEST_out = []
TESTV = []
TESTV_out = []
PATH = []
PATHV = []


def test2inputn(test, inp):
    #inp = np.zeros(len(test))
    #print(test)
    I = 0
    for t in test:
        inp[I] = t
        I += 1

    #return inp
    #display_test(inp, 1)

def display_input(inp):
    for i in range(size_input):
        print(str(inputn[i]))


def display_test(test, out):

	print("output :  " + str(out))
	for i in range(16):
		d = ''
		for j in range(16):
			d += str((test[i*16+j])) + " "
		print(d)

def display_weight_bias2():

    """print("Hiddenw 1")
    for i in range(size_hidden):
        for j in range(size_input):
            print(str(i) +":"+str(hiddenw[0][i][j]))

    print("Hidden b 1")
    for i in range(size_hidden):
        print(str(i) +":"+str(hiddenb[0][i]))"""

    print("Output b 1")
    for i in range(size_output):
        print(str(i) +":"+str(outputb[0][i]))

    print("Output res 1")
    for i in range(size_output):
        print(str(i) +":"+str(output[0][i]))



def display_weight_bias():

    print("Hiddenw 1")
    for i in range(size_hidden):
        for j in range(size_input):
            print(str(i) +":"+str(hiddenw[0][i][j]))
       
    print("Hiddenw 2")
    for i in range(size_hidden2):
        for j in range(size_hidden):
            print(str(i) +":"+str(hiddenw[1][i][j]))

    print("Hiddenw 3")
    for i in range(size_hidden3):
        for j in range(size_hidden2):
            print(str(i) +":"+str(hiddenw[2][i][j]))

    print("Hiddenw f")
    for i in range(size_hiddenf):
        for j in range(size_hidden3):
            print(str(i) +":"+str(hiddenw[3][i][j]))

    print("Outputw 1")
    for i in range(size_output):
        for j in range(size_hiddenf):
            print(str(i) +":"+str(outputw[0][i][j]))


    print("Hidden b 1")
    for i in range(size_hidden):
        print(str(i) +":"+str(hiddenb[0][i]))

    print("Hidden b 2")
    for i in range(size_hidden2):
        print(str(i) +":"+str(hiddenb[1][i]))

    print("Hidden b 3")
    for i in range(size_hidden3):
        print(str(i) +":"+str(hiddenb[2][i]))

    print("Hidden b f")
    for i in range(size_hiddenf):
        print(str(i) +":"+str(hiddenb[3][i]))

    print("Output b 1")
    for i in range(size_output):
        print(str(i) +":"+str(outputb[0][i]))

    print("Hidden res 1")
    for i in range(size_hidden):
        print(str(i) +":"+str(hidden[0][i]))

    print("Hidden res 2")
    for i in range(size_hidden2):
        print(str(i) +":"+str(hidden[1][i]))

    print("Hidden res 3")
    for i in range(size_hidden3):
        print(str(i) +":"+str(hidden[2][i]))

    print("Hidden res f")
    for i in range(size_hiddenf):
        print(str(i) +":"+str(hidden[3][i]))

    print("Output res 1")
    for i in range(size_output):
        print(str(i) +":"+str(output[0][i]))



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
    
def dRELU(x):
    if x > 0:
        return 1  
    else:
        return 0

def dRelu(x):
    if x > 0:
        return 1  
    else:
        return 0

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
            #print(( ri + hb[o] ) )
            out[I]=  RELU( ri + hb[o] ) 
        I += 1
    #print(out)
    #return out
    #
    #print("-----------------------------------")

def leaky_relu(x, alpha=0.01):
        return max(alpha*x, x)
def dleaky_relu(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def softmax(self, x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def calc_output_RELU(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I]=  leaky_relu( ri + hb[o] ) 
        I += 1

def calc_output_RELUF(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I] = leaky_relu(ri + hb[o])
        I += 1

def calc_output_RELUFSM(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I] = leaky_relu(ri + hb[o])
        I += 1

    # Appliquer la fonction softmax
    exp_out = np.exp(out)
    sum_exp_out = np.sum(exp_out)
    for o in range(size_out):
        out[o] = exp_out[o] / sum_exp_out

   
def calc_output_RELUD(inp, hw, hb, size_in, size_out, out, drop):
    #out = np.zeros(size_out)
    I = 0
    
    
    for o in range(size_out):
        if o in drop:
            out[I]=0 
            I += 1
            continue
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I]=  RELU( ri + hb[o] ) 
        I += 1

def calc_output_RELUFD(inp, hw, hb, size_in, size_out, out, drop):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            if i in drop:continue
            ri += inp[i] * hw[o,i]

        out[I] = RELU(ri + hb[o])
        I += 1

def calc_output_SG(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    
    
    for o in range(size_out):
        
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I]=  sigmoid( ri + hb[o] ) 
        I += 1

def calc_output_SGF(inp, hw, hb, size_in, size_out, out):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            
            ri += inp[i] * hw[o,i]

        out[I] = sigmoid(ri + hb[o])
        I += 1
          
def calc_output_SGD(inp, hw, hb, size_in, size_out, out, drop):
    #out = np.zeros(size_out)
    I = 0
    
    
    for o in range(size_out):
        if o in drop:
            out[I]=0 
            I += 1
            continue
        ri = 0
        for i in range(size_in):
            ri += inp[i] * hw[o,i]

        out[I]=  sigmoid( ri + hb[o] ) 
        I += 1

def calc_output_SGFD(inp, hw, hb, size_in, size_out, out, drop):
    #out = np.zeros(size_out)
    I = 0
    for o in range(size_out):
        ri = 0
        for i in range(size_in):
            if i in drop:continue
            ri += inp[i] * hw[o,i]

        out[I] = sigmoid(ri + hb[o])
        I += 1
        
    
   


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
                    
def hidden_backp71(alm1, cost, output, outputw, outputb, size_c, size_out, size_in):
        
        for c in range(size_c):
            #for o in range(size_out):
            d = derive(output[c])
            db = d * 2 * cost[c]
            outputb[c] -=  LR * db #+ random.random()
                            
            for i in range(size_in):
                
                
                dw = alm1[i] * d * 2 * cost[c]
                
                outputw[c,i] -= LR* dw
                
def hidden_backp72(alm1, cost, output, outputwp1,  size_outp1, outputw, outputb, size_c, size_out, size_in):
        
        #for c in range(size_c):
        for o in range(size_out):
            d = derive(output[o])
            db = d #* 2 * cost[c]
            outputb[o] -=  LR * db #+ random.random()
            
            etotal = 1
            #for wi in range(size_inp1):
            for wo in range(size_outp1):
            #
                etotal *=  outputwp1[wo][o]
            
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
                
                dw = alm1[i] * d * etotal # * cost[c]
                
                outputw[o,i] -= LR* dw                
            
def hidden_backp73(alm1, cost, output, outputwp1, size_outp1, outputwp2, size_outp2, outputw, outputb, size_c, size_out, size_in):
    
        etotal=1
        for o in range(size_outp2):
            for i in range(size_outp1):
            #
                etotal *=  outputwp2[o][i]
        
        #print(etotal)
        #for c in range(size_c):
        for o in range(size_out):
            d = derive(output[o])
            db = d #* 2 * cost[c]
            outputb[o] -=  LR * db #+ random.random()
            
            etotal2 = 1
            #for wi in range(size_inp1):
            for wo in range(size_outp1):
            #
                etotal2 *=  outputwp1[wo][o]
                
            
            
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
                
                dw = alm1[i] * d * etotal2 * etotal # * cost[c]
                
                outputw[o,i] -= LR* dw           
                
def backprop1(cost):
    for c in range(size_output):
            #for o in range(size_out):
            d = dRELU(output[0][c])
            db = d * 2 * cost[c]
            outputb[0][c] +=  LR * db #+ random.random()
                            
            for i in range(size_hiddenf):
                                
                dw = hidden[3][i] * d * 2.0 * cost[c]
                
                outputw[0][c,i] += LR* dw
                
def backprop2(cost, W):
    
    for o in range(size_hiddenf):
            d = dRELU(hidden[3][o])
            db = d #* 2 * cost[c]
            hiddenb[3][o] +=  LR * db #+ random.random()
            
            etotal = 1
            #for wi in range(size_inp1):
            for wo in range(size_output):
            #
                etotal *=  (outputw2[0][wo][o]+ random.random()*2.0) * 2 * cost[wo]
                
            #if W == 9:
             #   print(etotal)
            
            """nb = 0
            nbmax = 0
            if(size_hidden3 < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden3 * 0.25"""
            for i in range(size_hidden3):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[2][i] = 0.0"""
                
                dw = hidden[2][i] * d * etotal # * cost[c]
                #if W == 9:
                #    print(str(hidden[2][i] ) + " =" + str(hiddenw[3][o,i]) + " " + str(LR*dw))
                hiddenw[3][o,i] += LR* dw 
                #if W == 9:
                #    print(str(hidden[2][i] ) + " =" + str(hiddenw[3][o,i]) + " " + str(LR*dw))
                
def backprop3(cost, W):
    
    for o in range(size_hidden3):
            d = dRELU(hidden[2][o])
            db = d #* 2 * cost[c]
            hiddenb[2][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
            
                    etotal *=  (outputw2[0][wo][wi] + random.random()*2.0)* 2.0 *cost[wo]
            #print(etotal)    
            for wo in range(size_hiddenf):
            
                etotal *= ( hiddenw2[3][wo][o] + random.random()*2.0)
            
            #if W == 9:
             #   print(etotal) 
            """nb = 0
            nbmax = 0
            if(size_hidden2 < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden2 * 0.25"""
            for i in range(size_hidden2):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[1][i] = 0.0"""
                
                dw = hidden[1][i] * d * etotal # * cost[c]
                
                hiddenw[2][o,i] += LR* dw 
                
def backprop4(cost, W):
    
    for o in range(size_hidden2):
            d = dRELU(hidden[1][o])
            db = d #* 2 * cost[c]
            hiddenb[1][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
            
                    etotal *=  (outputw2[0][wo][wi]+ random.random()*2.0) * 2.0 *cost[wo]
            #print(etotal)   
            for wo in range(size_hiddenf):
                for wi in range(size_hidden3):
                    etotal *=  (hiddenw2[3][wo][wi] + random.random()*2.0)
            #print(etotal)    
            for wo in range(size_hidden3):
            
                etotal *= ( hiddenw2[2][wo][o] + random.random()*2.0)
                
            #if W == 9:
            #    print(etotal)
            
            """nb = 0
            nbmax = 0
            if(size_hidden < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden * 0.25"""
            for i in range(size_hidden):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[0][i] = 0.0"""
                
                dw = hidden[0][i] * d * etotal # * cost[c]
                
                hiddenw[1][o,i] += LR* dw 
                
def backprop5(cost, W):
    
    #display_test(inputn, 2)
    
    for o in range(size_hidden):
            d = dRELU(hidden[0][o])
            #print(hidden)
            db = d #* 2 * cost[c]
            hiddenb[0][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
                    #print(outputw2[0][wo][wi])
                    #print(cost[wo])
                    etotal *=  (outputw2[0][wo][wi]+ random.random()*2.0) * 2.0 *cost[wo]
            #print(etotal)   
            for wo in range(size_hiddenf):
                for wi in range(size_hidden3):
                    #print(hiddenw2[3][wo][wi] )
                    etotal *=  (hiddenw2[3][wo][wi] + random.random()*2.0)
            #print(etotal)    
            for wo in range(size_hidden3):
                for wi in range(size_hidden2):
                    #print(hiddenw2[2][wo][wi] )
                    etotal *=  (hiddenw2[2][wo][wi] + random.random()*2.0)
            #print(etotal)   
            for wo in range(size_hidden2):
                #print(hiddenw2[1][wo][wi] )
                etotal *=  (hiddenw2[1][wo][o] + random.random()*2.0)
                
            #print(etotal)
            #print(hiddenw2)
            
            
            nb = 0
            nbmax = 0
            if(size_input < 4):
                nbmax = 1.0
            else:
                nbmax = size_input * 0.25
            for i in range(size_input):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    inputn[i] = 0.0"""
                
                dw = inputn[i] * d * etotal # * cost[c]
                #if W == 9:
                 #   print("--------dw:" + str(LR*dw)  + " inputn: " + str(inputn[i]) + " d: " + str(d)  + " etotal:" + str(etotal))
                #if W == 9:
                #    print(hiddenw[0][o,i])
                hiddenw[0][o,i] += LR* dw 
                #if W == 9:
                #    print(hiddenw[0][o,i])
            

def Backpropagation512(cost):

    dw = 0;
    for o in range(size_output):
        for i in range(size_hidden):
            outputw[0][o, i] -= LR * cost[o] * hidden[0][i]

    for o in range(size_output):
        outputb[0][o] -=  LR * cost[o]

    print("end output back")

    #hiddenf
    for o in range(size_output):
        for f in range(size_hidden):
            for i in range(size_input):
                    hiddenw[0][f,i] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[0][f]) * inputn[i]

    for o in range(size_output):
        for f in range(size_hidden):
            hiddenb[0][f] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[0][f]) 

    print("end hiddenf back")

def Backpropagation512sig(cost):

    dw = 0;
    for o in range(size_output):
        for i in range(size_hidden):
            outputw[0][o, i] -= LR * cost[o] * hidden[0][i]

    for o in range(size_output):
        outputb[0][o] -=  LR * cost[o]

    print("end output back")

    #hiddenf
    for o in range(size_output):
        for f in range(size_hidden):
            for i in range(size_input):
                    hiddenw[0][f,i] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[0][f]) * inputn[i]

    for o in range(size_output):
        for f in range(size_hidden):
            hiddenb[0][f] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[0][f]) 

    print("end hiddenf back")



def Backpropagation(num_output, cost):

    dw = 0;
    for i in range(size_hiddenf):
        outputw[0][num_output, i] -= LR * cost * hidden[3][i]

    outputb[0][num_output] -=  LR * cost

    print("end output back")

    #hiddenf
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
                hiddenw[3][f,i] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hidden[2][i]

    for f in range(size_hiddenf):
        hiddenb[3][f] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) 

    print("end hiddenf back")

    #hidden3
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                hiddenw[2][i, j] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hidden[1][j]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
                hiddenb[2][i] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) 

    print("end hidden3 back")

    #hidden2
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    hiddenw[1][j, k] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hidden[0][k]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                hiddenb[1][j] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j])

    print("end hidden2 back")
    #hidden
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    for l in range(size_input):
                        #print("f " + str(f) + " "+ str(i) + " "+ str(j) + " "+ str(k) + " " + str(l) + " "+ str(size_hiddenf))
                        hiddenw[0][k, l] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hiddenw[1][j, k] * dRelu(hidden[0][k]) * inputn[l]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    hiddenb[0][k] -= LR * cost * outputw[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hiddenw[1][j, k] * dRelu(hidden[0][k])

    print("end hidden back")

def BackpropagationRM(cost):

    dw = 0;
    for o in range(size_output):
        for i in range(size_hiddenf):
            outputw[0][o, i] -= LR * cost[o] * hidden[3][i]

    for o in range(size_output):
        outputb[0][o] -=  LRB * cost[o]

    print("end output back")

    #hiddenf
    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                    hiddenw[3][f,i] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hidden[2][i]

    for o in range(size_output):
        for f in range(size_hiddenf):
            hiddenb[3][f] -= LRB * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) 

    print("end hiddenf back")

    #hidden3
    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                for j in range(size_hidden2):
                    hiddenw[2][i, j] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hidden[1][j]

    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                    hiddenb[2][i] -= LRB * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) 

    print("end hidden3 back")

    #hidden2
    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                for j in range(size_hidden2):
                    for k in range(size_hidden):
                        hiddenw[1][j, k] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hidden[0][k]

    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                for j in range(size_hidden2):
                    hiddenb[1][j] -= LRB * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j])

    print("end hidden2 back")
    #hidden
    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                for j in range(size_hidden2):
                    for k in range(size_hidden):
                        for l in range(size_input):
                            #print("f " + str(f) + " "+ str(i) + " "+ str(j) + " "+ str(k) + " " + str(l) + " "+ str(size_hiddenf))
                            hiddenw[0][k, l] -= LR * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hiddenw[1][j, k] * dRelu(hidden[0][k]) * inputn[l]

    for o in range(size_output):
        for f in range(size_hiddenf):
            for i in range(size_hidden3):
                for j in range(size_hidden2):
                    for k in range(size_hidden):
                        hiddenb[0][k] -= LRB * cost[o] * outputw[0][o, f] * dRelu(hidden[3][f]) * hiddenw[3][f,i] * dRelu(hidden[2][i]) * hiddenw[2][i, j] * dRelu(hidden[1][j]) * hiddenw[1][j, k] * dRelu(hidden[0][k])

    print("end hidden back")


def Backpropagation2(num_output, cost):

    dw = 0;
    for i in range(size_hiddenf):
        outputw[0][num_output, i] -= LR * cost * hidden[3][i]

    outputb[0][num_output] -=  LR * cost

    print("end output back")

    #hiddenf
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
                hiddenw[3][f,i] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hidden[2][i]

    for f in range(size_hiddenf):
        hiddenb[3][f] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) 

    print("end hiddenf back")

    #hidden3
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                hiddenw[2][i, j] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) * hidden[1][j]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
                hiddenb[2][i] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) 

    print("end hidden3 back")

    #hidden2
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    hiddenw[1][j, k] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) * hiddenw2[2][i, j] * dRelu(hidden[1][j]) * hidden[0][k]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                hiddenb[1][j] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) * hiddenw2[2][i, j] * dRelu(hidden[1][j])

    print("end hidden2 back")
    #hidden
    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    for l in range(size_input):
                        #print("f " + str(f) + " "+ str(i) + " "+ str(j) + " "+ str(k) + " " + str(l) + " "+ str(size_hiddenf))
                        hiddenw[0][k, l] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) * hiddenw2[2][i, j] * dRelu(hidden[1][j]) * hiddenw2[1][j, k] * dRelu(hidden[0][k]) * inputn[l]

    for f in range(size_hiddenf):
        for i in range(size_hidden3):
            for j in range(size_hidden2):
                for k in range(size_hidden):
                    hiddenb[0][k] -= LR * cost * outputw2[0][num_output, f] * dRelu(hidden[3][f]) * hiddenw2[3][f,i] * dRelu(hidden[2][i]) * hiddenw2[2][i, j] * dRelu(hidden[1][j]) * hiddenw2[1][j, k] * dRelu(hidden[0][k])

    print("end hidden back")




def backprop12(cost):
    for c in range(size_output):
            #for o in range(size_out):
            d = dRELU(output[0][c])
            db = d * 2 * cost[c]
            outputb[0][c] +=  LR * db #+ random.random()
                            
            for i in range(size_hiddenf):
                                
                dw = hidden[3][i] * d * 2.0 * cost[c]
                
                outputw[0][c,i] += LR* dw
                
def backprop22(cost, W):
    
    for o in range(size_hiddenf):
            d = dRELU(hidden[3][o])
            db = d * 2 * cost[0] * 2 * cost[1]
            hiddenb[3][o] +=  LR * db #+ random.random()
            
            etotal = 1
            #for wi in range(size_inp1):
            for wo in range(size_output):
            #
                etotal *=  (outputw2[0][wo][o]+ random.random()*2.0) * 2 * cost[wo]
                
            #if W == 9:
             #   print(etotal)
            
            """nb = 0
            nbmax = 0
            if(size_hidden3 < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden3 * 0.25"""
            for i in range(size_hidden3):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[2][i] = 0.0"""
                
                dw = hidden[2][i] * d * etotal # * cost[c]
                #if W == 9:
                #    print(str(hidden[2][i] ) + " =" + str(hiddenw[3][o,i]) + " " + str(LR*dw))
                hiddenw[3][o,i] += LR* dw 
                #if W == 9:
                #    print(str(hidden[2][i] ) + " =" + str(hiddenw[3][o,i]) + " " + str(LR*dw))
                
def backprop32(cost, W):
    
    for o in range(size_hidden3):
            d = dRELU(hidden[2][o])
            db = d * 2 * cost[0] * 2 * cost[1]
            hiddenb[2][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
            
                    etotal *=  (outputw2[0][wo][wi] + random.random()*2.0)* 2.0 *cost[wo]
            #print(etotal)    
            for wo in range(size_hiddenf):
            
                etotal *= ( hiddenw2[3][wo][o] + random.random()*2.0)
            
            #if W == 9:
             #   print(etotal) 
            """nb = 0
            nbmax = 0
            if(size_hidden2 < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden2 * 0.25"""
            for i in range(size_hidden2):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[1][i] = 0.0"""
                
                dw = hidden[1][i] * d * etotal # * cost[c]
                
                hiddenw[2][o,i] += LR* dw 
                
def backprop42(cost, W):
    
    for o in range(size_hidden2):
            d = dRELU(hidden[1][o])
            db = d * 2 * cost[0] * 2 * cost[1]
            hiddenb[1][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
            
                    etotal *=  (outputw2[0][wo][wi]+ random.random()*2.0) * 2.0 *cost[wo]
            #print(etotal)   
            for wo in range(size_hiddenf):
                for wi in range(size_hidden3):
                    etotal *=  (hiddenw2[3][wo][wi] + random.random()*2.0)
            #print(etotal)    
            for wo in range(size_hidden3):
            
                etotal *= ( hiddenw2[2][wo][o] + random.random()*2.0)
                
            #if W == 9:
            #    print(etotal)
            
            """nb = 0
            nbmax = 0
            if(size_hidden < 4):
                nbmax = 1.0
            else:
                nbmax = size_hidden * 0.25"""
            for i in range(size_hidden):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    hidden[0][i] = 0.0"""
                
                dw = hidden[0][i] * d * etotal # * cost[c]
                
                hiddenw[1][o,i] += LR* dw 
                
def backprop52(cost, W):
    
    #display_test(inputn, 2)
    
    for o in range(size_hidden):
            d = dRELU(hidden[0][o])
            #print(hidden)
            db = d * 2 * cost[0] * 2 * cost[1]
            hiddenb[0][o] +=  LR * db #+ random.random()
            
            etotal = 1
            for wo in range(size_output):
                for wi in range(size_hiddenf):
                    #print(outputw2[0][wo][wi])
                    #print(cost[wo])
                    etotal *=  (outputw2[0][wo][wi]+ random.random()*2.0) * 2.0 *cost[wo]
            #print(etotal)   
            for wo in range(size_hiddenf):
                for wi in range(size_hidden3):
                    #print(hiddenw2[3][wo][wi] )
                    etotal *=  (hiddenw2[3][wo][wi] + random.random()*2.0)
            #print(etotal)    
            for wo in range(size_hidden3):
                for wi in range(size_hidden2):
                    #print(hiddenw2[2][wo][wi] )
                    etotal *=  (hiddenw2[2][wo][wi] + random.random()*2.0)
            #print(etotal)   
            for wo in range(size_hidden2):
                #print(hiddenw2[1][wo][wi] )
                etotal *=  (hiddenw2[1][wo][o] + random.random()*2.0)
                
            #print(etotal)
            #print(hiddenw2)
            
            
            nb = 0
            nbmax = 0
            if(size_input < 4):
                nbmax = 1.0
            else:
                nbmax = size_input * 0.25
            for i in range(size_input):
                """r = random.random()
                if (r <= 0.25) and (nb < nbmax):
                    nb += 1
                    inputn[i] = 0.0"""
                
                dw = inputn[i] * d * etotal # * cost[c]
                #if W == 9:
                 #   print("--------dw:" + str(LR*dw)  + " inputn: " + str(inputn[i]) + " d: " + str(d)  + " etotal:" + str(etotal))
                #if W == 9:
                #    print(hiddenw[0][o,i])
                hiddenw[0][o,i] += LR* dw 
                #if W == 9:
                #    print(hiddenw[0][o,i])
                    
                    
                    
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

def Copy(hidden, hidden2, hiddenw, hiddenw2, hiddenb, hiddenb2, output, output2, outputw, outputw2, outputb, outputb2):
    
    hidden2[0] = hidden[0].copy()
    hidden2[1] = hidden[1].copy()
    hidden2[2] = hidden[2].copy()
    hidden2[3] = hidden[3].copy()
    
    #print(hidden2)
    #print(hidden)
    
    hiddenw2[0] = hiddenw[0].copy()
    hiddenw2[1] = hiddenw[1].copy()
    hiddenw2[2] = hiddenw[2].copy()
    hiddenw2[3] = hiddenw[3].copy()
    
    #print(hiddenw2[0][0][0])
    
    hiddenb2[0] = hiddenb[0].copy()
    hiddenb2[1] = hiddenb[1].copy()
    hiddenb2[2] = hiddenb[2].copy()
    hiddenb2[3] = hiddenb[3].copy()
    
    output2[0] = output[0].copy()
    outputw2[0] = outputw[0].copy()
    outputb2[0] = outputb[0].copy()

def Afficher_images(path_dir, valid, nb, score, pct):
    
    I = 1
    #ListeFichiers = os.listdir(path_dir)
    plt.figure(figsize=(15.5, 14.0), dpi=100)
   
    
    plt.subplot(2,1,1)
    Img_Pil = img.load_img(path_dir[0], target_size=(150, 150))
    plt.imshow(Img_Pil)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(score) + "/" + str(nb) + " soit " + str(pct) + "%" , size=20, color="Blue")
    
    for NoImg in range(nb):
               
        Img_Pil = img.load_img(path_dir[NoImg], target_size=(150, 150))
        #Img_Array = img.img_to_array(Img_Pil)/255
        #Img_List = np.expand_dims(Img_Array, axis=0)
        
        plt.subplot(10,10,NoImg+1)
        plt.imshow(Img_Pil)
        plt.xticks([])
        plt.yticks([])
        
        if valid[NoImg]:
            plt.title('Bien ' + str(I) + '/' + str(nb),  size=10, color="Green")
            #plt.title(str(I), pad=1, size=10, color="Blue")
            I += 1
        else:
            plt.title('Mal classe',  size=10, color="Red")
                    
    plt.show()
    
def Afficher_images2(path_dir, valid, nb, score, pct):
    
    I = 1
    #ListeFichiers = os.listdir(path_dir)
    plt.figure(figsize=(15.5, 14.0), dpi=100)
   
    
    plt.subplot(2,1,1)
    Img_Pil = img.load_img(path_dir[0], target_size=(50, 50))
    plt.imshow(Img_Pil)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(score) + "/" + str(nb) + " soit " + str(pct) + "%" , size=10, color="Blue")
    
    for NoImg in range(nb):
               
        Img_Pil = img.load_img(path_dir[NoImg], target_size=(50, 50))
        #Img_Array = img.img_to_array(Img_Pil)/255
        #Img_List = np.expand_dims(Img_Array, axis=0)
        
        plt.subplot(10,int(math.ceil(nb / 10)),NoImg+1)
        plt.imshow(Img_Pil)
        plt.xticks([])
        plt.yticks([])
        
        if valid[NoImg]:
            plt.title('Bien ' + str(I) + '/' + str(nb),  size=7, color="Green")
            #plt.title(str(I), pad=1, size=10, color="Blue")
            I += 1
        else:
            plt.title('Mal classe',  size=7, color="Red")
                    
    plt.show()

def Validation(TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2):
    
    score = 0.0
    score2 = 0.0
    score3 = 0
    score4 = 0
    nbcat = 50.0
    nbdog = 50.0

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
    I2=0
    lotsz = 10
    ilot = 0
    alt = True
    IV = 0

    while not stop:
        print("I: " + str(I) + " " + str(I2))       

        if alt:
            test2inputn(TESTV[I], inputn)
        else:
            test2inputn(TESTV2[I2], inputn)

    
        #for i in range(size_input):
            #inputn[i] /= 255.0
            #inputn[i] = ((inputn[i])*1000+1) / 10000.0;

        #BN(inputn, size_input, NORM, inputn)
        #display_test(inputn, TESTV_out[I])
            #change_in = False

        

        #calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        #calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])

        calc_output_SG(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        calc_output_SGF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
        
        """calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        calc_output_RELUF(hidden[3], outputw[0], outputb[0], size_hiddenf, size_output_bias, output[0])"""

        #calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        #calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
    

        #print(str(I) + " : " + str(output[0]))
        
        
        maxi = -10000000.0
        resom = -1
        sum = 0
        for o in range(size_output):
            sum += output[0][o]
            print("resp " + str(o) + " " + str(output[0][o]))
            if output[0][o] > maxi:
                resom = o+1
                maxi = output[0][o]

        #abs(output[0][0] - 2.0)

        #print("resp " + str(0) + " " + str(output[0][0]))
        #if(output[0][0] >= 0.0):
        #    resom = 1
        #elif output[0][0] < 0.0:
        #    resom = 2

        
        name = ""
        if alt:
            print(PATHV[I]);
            print("chihuahua")
            name = "chihuahua"
        else:
            print(PATHV2[I2]);
            name="muffin"
            print("muffin")
            
        #print(str(I) + " : " + name)
        
        if resom == 1:
            #print("cat")
            if alt:
                score += 1
                print("juste")
                
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
            pct = (output[0][0] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 2:
            #print("dog")
            if not alt:
                print("juste")
                
                score += 1
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
            pct = (output[0][1] / sum) * 100
            print("pct : " + str(pct) + "%")

        IV +=1
        if alt:
            I += 1
        else:
            I2+=1

        ilot+=1
        if ilot == lotsz:
            alt = not alt
            ilot = 0

        




        change_in = True

        cnt = 0	
        E = 0		

        if I == len(TESTV) and I2 == len(TESTV2):
            I = 0
            W+=1
            stop = True
            #if W == WMAX:
            #    stop = True
                

    pct = float(score/(nbcat+nbdog))*100.0
    print("Algo Score:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
    
    print("---------------------------------------------------------------------------------")

    return score, pct



def Validation3(TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3):
    
    score = 0.0
    score2 = 0.0
    score3 = 0
    score4 = 0
    nbcat = 50.0
    nbdog = 50.0

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
    I2=0
    I3=0
    lotsz = 32
    ilot = 0
    alt = 1
    IV = 0

    while not stop:
        print("I: " + str(I) + " " + str(I2))       

        if alt==1:
            test2inputn(TESTV[I], inputn)
        elif alt ==2:
            test2inputn(TESTV2[I2], inputn)

        else:
            test2inputn(TESTV3[I3], inputn)
    
        #for i in range(size_input):
            #inputn[i] /= 255.0
            #inputn[i] = ((inputn[i])*1000+1) / 10000.0;

        #BN(inputn, size_input, NORM, inputn)
        #display_test(inputn, TESTV_out[I])
            #change_in = False

        

        calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])

        #calc_output_SG(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_SGF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
        
        """calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        calc_output_RELUF(hidden[3], outputw[0], outputb[0], size_hiddenf, size_output_bias, output[0])"""

        #calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        #calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
    

        #print(str(I) + " : " + str(output[0]))
        
        
        maxi = -10000000.0
        resom = -1
        sum = 0
        for o in range(size_output):
            sum += output[0][o]
            print("resp " + str(o) + " " + str(output[0][o]))
            if output[0][o] > maxi:
                resom = o+1
                maxi = output[0][o]

        #abs(output[0][0] - 2.0)

        #print("resp " + str(0) + " " + str(output[0][0]))
        #if(output[0][0] >= 0.0):
        #    resom = 1
        #elif output[0][0] < 0.0:
        #    resom = 2

        
        name = ""
        if alt == 1:
            print(PATHV[I]);
            print("chihuahua")
            name = "chihuahua"
        elif alt == 2:
            print(PATHV2[I2]);
            name="fleur"
            print("fleur")
        else:
            print(PATHV3[I3]);
            name="muffin"
            print("muffin")
            
        #print(str(I) + " : " + name)
        
        if resom == 1:
            #print("cat")
            if alt == 1:
                score += 1
                print("juste")
                
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
            pct = (output[0][0] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 2:
            #print("dog")
            if alt == 2:
                print("juste")
                
                score += 1
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
            pct = (output[0][1] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 3:
            #print("dog")
            if alt == 3:
                print("juste")
                
                score += 1
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
            pct = (output[0][2] / sum) * 100
            print("pct : " + str(pct) + "%")

        IV +=1
        if alt == 1:
            I += 1
        elif alt == 2:
            I2+=1
        elif alt == 3:
            I3+=1

        ilot+=1
        if ilot == lotsz:
            alt += 1
            if alt == 4:
                alt = 1
            ilot = 0

        




        change_in = True

        cnt = 0	
        E = 0		

        if I == len(TESTV) and I2 == len(TESTV2) and I3 == len(TESTV3):
            I = 0
            W+=1
            stop = True
            #if W == WMAX:
            #    stop = True
                

    pct = float(score/150.0)*100.0
    print("Algo Score:" + str(score) + "/" + str(150) + " soit " + str(pct) + "%")
    
    print("---------------------------------------------------------------------------------")

    return score, pct


def ValidationNN(nn, TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2, TESTV3, TESTV_out3, PATHV3, res):
    
    score = 0.0
    score2 = 0.0
    score3 = 0
    score4 = 0
    nbcat = 50.0
    nbdog = 50.0

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
    I2=0
    I3=0
    lotsz = 32
    ilot = 0
    alt = 1
    IV = 0

    nn.init_F1()

    while not stop:
        print("I: " + str(I) + " " + str(I2) + " " + str(I3))       

        if alt == 1:
            nn.build(np.array(TESTV[I]))
                      
        elif alt == 2:
            nn.build(np.array(TESTV2[I2]))
           
        else:
            nn.build(np.array(TESTV3[I3]))

        nn.BN(1.0)

        #for i in range(size_input):
            #inputn[i] /= 255.0
            #inputn[i] = ((inputn[i])*1000+1) / 10000.0;

        #BN(inputn, size_input, NORM, inputn)
        #display_test(inputn, TESTV_out[I])
            #change_in = False

        nn.predict_softmax()

        #calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        #calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])

        #calc_output_SG(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_SGF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
        
        """calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        calc_output_RELUF(hidden[3], outputw[0], outputb[0], size_hiddenf, size_output_bias, output[0])"""

        #calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
        #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
        #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
        #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
        #calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])
    

        #print(str(I) + " : " + str(output[0]))
        
        
        maxi = -10000000.0
        resom = -1
        sum = 0
        for o in range(nn.size_output):
            sum += nn.layero[o]
            print("resp " + str(o) + " " + str(nn.layero[o]))
            if nn.layero[o] > maxi:
                resom = o+1
                maxi = nn.layero[o]

        
        
        name = ""
        if alt == 1:
            print(PATHV[I]);
            print("chihuahua")
            name = "chihuahua"
            if resom != alt:
                nn.fn += 1
        elif alt == 2:
            print(PATHV2[I2]);
            name="fleur"
            print("fleur")
            if resom != alt:
                nn.fn += 1
        elif alt == 3:
            print(PATHV3[I3]);
            name="muffin"
            print("muffin")
            if resom != alt:
                nn.fn += 1
            
        #print(str(I) + " : " + name)

        
        
        if resom == 1:
            #print("cat")
            if alt == 1:
                score += 1
                res[0] +=1
                print("juste")
                nn.tp+=1
                valid[IV] = True
            else:
                valid[IV] = False
                print("faux")
                nn.fp += 1
            pct = (nn.layero[0] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 2:
            #print("dog")
            if alt == 2:
                print("juste")
                nn.tp+=1
                res[1] +=1
                score += 1
                valid[IV] = True
            else:
                valid[IV] = False
                nn.fp += 1
                print("faux")
            pct = (nn.layero[1] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 3:
            #print("dog")
            if alt == 3:
                print("juste")
                nn.tp+=1
                res[2] +=1
                score += 1
                valid[IV] = True
            else:
                valid[IV] = False
                nn.fp += 1
                print("faux")
            pct = (nn.layero[2] / sum) * 100
            print("pct : " + str(pct) + "%")
        elif resom == 4:
            print("autre")
            valid[IV] = False

        IV +=1
        if alt == 1:
            I += 1
        elif alt == 2:
            I2+=1
        elif alt == 3:
            I3+=1

        ilot+=1
        if ilot == lotsz:
            alt += 1
            if alt == 4:
                alt = 1
            ilot = 0

        




        change_in = True

        cnt = 0	
        E = 0		

        if I == len(TESTV) and I2 == len(TESTV2) and I3 == len(TESTV3):
            I = 0
            W+=1
            stop = True
            #if W == WMAX:
            #    stop = True
                

    pct = float(score/192.0)*100.0
    print("Algo Score:" + str(score) + "/" + str(192) + " soit " + str(pct) + "%")
    print("F1 score : " + str(nn.F1_score()))
    print("---------------------------------------------------------------------------------")

    return score, pct



























