

import random
import math


inputn = []
size_input = 256
hidden = []
hiddenw = []
hiddenb = []
nb_hidden = 3
size_hidden = 64
size_hidden2 = 64
size_hidden3 = 64#30#100#44
sz_hidden = [size_hidden, size_hidden2]#no 
size_hidden_weight = [size_input, size_hidden]#no
size_hidden_bias = [size_hidden, size_hidden2, size_hidden3]#use

output = []
outputw = []
outputb = []
size_output = 2
size_output_weight = size_hidden2
size_output_bias = size_output
LR = 0.0009

TEST = []
TEST_out = []
TESTV = []
TESTV_out = []


def test_proc(test, out):
	
	TEST.append(test)
	TEST_out.append(out)
	

def display_test(test, out):

	print("output :  " + str(out))
	for i in range(16):
		d = ''
		for j in range(16):
			d += str((test[i*16+j])) + " "
		print(d)
	

def display_TEST():
	for i in range(len(TEST)):
		display_test(TEST[i], TEST_out[i])
		print("")

def test2inputn(test):
	inp = []
	for t in test:
		inp.append(t)
	return inp


def display_inputn(inp):
	print(inp)
	
"""
test = [0, 0, 1, 0, 0,
	0, 0, 1, 0, 0,
	1, 1, 1, 1, 1,
	0, 0, 1, 0, 0,
	0, 0, 1, 0, 0]

test_proc(test, 1)

test = [0, 0, 1, 0, 0,
	0, 0, 1, 0, 0,
	0, 1, 1, 1, 1,
	0, 0, 1, 0, 0,
	0, 0, 1, 0, 0]

test_proc(test, 1)

test = [0, 0, 1, 0, 0,
	0, 0, 1, 0, 0,
	1, 1, 1, 1, 0,
	0, 0, 1, 0, 0,
	0, 0, 1, 0, 0]

test_proc(test, 1)

test = [0, 0, 0, 0, 0,
	0, 0, 1, 0, 0,
	1, 1, 1, 1, 1,
	0, 0, 1, 0, 0,
	0, 0, 1, 0, 0]

test_proc(test, 1)

test = [0, 0, 1, 0, 0,
	0, 0, 1, 0, 0,
	1, 1, 1, 1, 1,
	0, 0, 1, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 1)

test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	1, 1, 1, 1, 1,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)

test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 1, 1, 1, 1,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)

test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	1, 1, 1, 1, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)

test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0.5, 1, 1, 1,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)


test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	1, 1, 1, 0.5, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)

test = [0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	1, 1, 1, 0.5, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0]

test_proc(test, 2)

test = [0, 0, 0, 0, 0,
	1, 1, 0, 0, 0,
	0, 0, 1, 0, 0,
	0, 0, 0, 1, 1,
	0, 0, 0, 0, 0]

test_proc(test, 2)
"""
#********************************************************************************

def display_hidden(hid):
	for h in hid:
		print("size=" + str(len(h)))
		print(h)

def display_hidden_w(hid):
	for h in hid:
		print("size=" + str(len(h)))
		for w in h:
			print("----size=" + str(len(w)))
			#print(w)
			

def random_hidden(hid, szo, szw):

		
	for no in range(szo):
		wl = []
		for nw in range(szw):
			wl.append(random.random()*1-0.5)#(48.0/float(size_input))-(24/float(size_input)))

		hid.append(wl)


	

def random_hidden_bias_RELU(hid, size):

	for n in range(nb_hidden):
		
		w = []
		for i in range(size[n]):
			ns = random.random()
			nb = random.random()
			if ns > 5:
				w.append(nb)
			else:
				w.append(-nb)
		hid.append(w)

def random_hidden_bias_sigmoid(hid, size):

	for n in range(nb_hidden):
		
		w = []
		for i in range(size[n]):
			w.append(random.random()*0.02-0.01)#*(48.0/float(size_input))-(24/float(size_input)))

		hid.append(w)

def random_output_w(hid, szo, szw):
		
		
	for no in range(szo):
		wl = []
		for nw in range(szw):
			wl.append(random.random()*1-0.5)#*(48.0/float(size_input))-(24/float(size_input)) )

		hid.append(wl)

def random_output_bias_sigmoid(hid, size):

	
	w = []
	for i in range(size):
		w.append(random.random()*0.02-0.01)

	hid.append(w)

def display_hiddenw():
	print("hidden weight")
	display_hidden_w(hiddenw)

def random_weight():
	hiddenw.append([])	
	#out in
	random_hidden(hiddenw[0], size_hidden, size_input)
	hiddenw.append([])
	random_hidden(hiddenw[1], size_hidden2, size_hidden)
	hiddenw.append([])
	random_hidden(hiddenw[2], size_hidden3, size_hidden2)
	



def display_hiddenb():
	print("hidden bias")
	display_hidden(hiddenb)

def random_bias(type):
	if type == "RELU":
		random_hidden_bias_RELU(hiddenb, size_hidden_bias)
	elif type == "sigmoid":
		random_hidden_bias_sigmoid(hiddenb, size_hidden_bias)

def display_hiddeneuron():
	print("hidden neuron")
	display_hidden(hidden)

def random_outputw():
	outputw.append([])
	random_output_w(outputw[0], size_output, size_hidden3)

def display_outputw():
	for h in outputw:
		print("size=" + str(len(h)))
		for w in h:
			print("----size=" + str(len(w)))
			print(w)

def random_outputb():
	random_output_bias_sigmoid(outputb, size_output) 

def display_outputb():
	print("output bias")
	display_hidden(outputb)

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

#***********************************************************************************

def calc_output_sigmoid(inp, hw, hb, size_in, size_out):
	out = []
	
	for o in range(size_out):
 		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[i]
			
		out.append( sigmoid( ri + hb[o] ) )


	return out

def calc_output_RELU(inp, hw, hb, size_in, size_out):
	out = []
	
	for o in range(size_out):
 		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[i]
			
		out.append( RELU( ri + hb[o] ) )


	return out

def calc_output_sa(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]
			
		r = random.random() * 10
		
		out.append(  ri + hb[o]  )

	return out

def calc_output_sigmoid2(inp, hw, hb, size_in, size_out):
	out = []
	nbmax = 0
	if(size_out < 4):
		nbmax = 1.0
	else:
		nbmax = size_out * 0.9
	
        nb = 0
        for o in range(size_out):
            ri = 0
            for i in range(size_in):
                ri += inp[i] * hw[o][i]
                
            """r = random.random() * 10
            
            if (r < 5) and (nb < nbmax):
                out.append(0)
                nb += 1
            else:"""
            out.append( sigmoid( ri + hb[o] ) )#

	return out

def calc_output_sigmoid3(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]
			
		out.append( sigmoid( ri + hb[o] ) )#

	return out

def calc_output_softmax2(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]
			
		out.append( softmax( ri + hb[o] ) )

	return out

# [ [16[25]], [16[16]] ]
def calc_output_RELU2(inp, hw, hb, size_in, size_out):
        out = []
        nbmax = 0
        if(size_out < 4):
            nbmax = 1.0
        else:
            nbmax = size_out * 0.25
        nb = 0
        for o in range(size_out):
            ri = 0
            for i in range(size_in):
                ri += inp[i] * hw[o][i]

            r = random.random() * 10
            
            if (r < 2.5) and (nb < nbmax):
                out.append(0)
                nb += 1
            else:
                out.append( RELU( ri + hb[o] ) )#
            
        return out

def calc_output_RELU3(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]

		out.append( RELU( ri + hb[o] ) )

	return out

def calc_output_RELUF(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]

		out.append( ( ri + hb[o] ) )

	return out

def calc_output_tanh2(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]

		out.append( tanh( ri + hb[o] ) )

	return out

#***********************************************************************************
#backpropagation


def Error(target, out):
	return ((target * out) * (target * out)) * 0.5

"""
dEtot/dw5 = dEtot/dout1 * dout1/dnet1 * dnet1/dw5


dEtot/dout1 = -(target1 - out1)
dout1/dnet1 = dout1 x (1 - dout1)
dnet1/dw5 = dout1

w5' = w5 - lr x dEtot/dw5

dE1/douth1 = dE1/dnet1 * dnet1/douth1
		0.1384 * 0.4(w5)

"""

def derive(x):
	return x * (1 - x)


def hidden_backp(output, Loss, outputb, outputw, layer, size_out, size_in, JM, JB, first):

	for o in range(size_out):
		d = derive(output[o])
		db = 0
		"""if first:
			db = LR * Loss[o]
		else:"""
		db = LR * Loss[o] #* d
		#print("db:" + str(db) + " b:" + str(outputb[o]))
		outputb[o] +=  db#*d #LR * JB 
		for i in range(size_in):
			#warg = 0.95 * warg + (1 - 0.95) * (derive(Loss[o])/derive(outputw[o][i])) *(derive(Loss[o])/derive(outputw[o][i]))
			#LRP = LR / math.sqrt(warg + 0.1)
			#print("w:" + str(outputw[o][i]))
			#x = layer[i] * outputw[o][i] + outputb[o]
			#n =  0.3333333 * (1 + math.exp(-x)) - 1
			#n =  0.3333333 * RELU(x) - 1
			#n = (n -  outputb[o]) * layer[i]
			#db * layer[i] *
			"""g = 0
			for W in range(100):
				g = 0.95 * g + (1 - 0.95) * (Loss[o] * d) * (Loss[o] * d)
				outputw[o][i] -= db / (math.sqrt(g + 0.0000001))"""
			#print(outputw[o][i])
			outputw[o][i] +=   db * layer[i]
			#outputw[o][i] -= LR * JM
			#outputw[o][i] -= LRP * (derive(Loss[o]) / derive(outputw[o][i]))

	#return warg

def hidden_backp2(output, Loss, outputb, outputw, layer, size_out, size_in):

	for o in range(size_out):
		d = derive(output[o])
		db = LR * Loss[o] * d
		outputb[o] +=  db 
		for i in range(size_in):
			outputw[o][i] += db * layer[o]

def hidden_backp3(output, Loss, LRL,  outputb, outputw, layer, size_l, size_out, size_in):

	for l in range(size_l):
		db = LRL[l] * Loss[l] 
		#for T in range(2):
		for o in range(size_out):
			#d = derive(output[o])
		
			outputb[o] +=  db 
			for i in range(size_in):
				"""g = s = 0
				for W in range(10):
					g = 0.95 * g + (1 - 0.95) * (Loss[l] ) * (Loss[l])
					outputw[o][i] -= db / (math.sqrt(g + 0.0000001)) * layer[i]"""
				outputw[o][i] += db 
				
def hidden_backp4(alm1, cost, output, outputw, outputb, size_c, size_out, size_in):
    
        for c in range(size_c):
            for o in range(size_out):
                d = derive(output[o])
                db = d * 2 * cost[c]
                outputb[o] +=  LR * db #+ random.random()
                
                nb = 0
                nbmax = 0
                if(size_in < 4):
                    nbmax = 1.0
                else:
                    nbmax = size_in * 0.25
                for i in range(size_in):
                    r = random.random() * 10
                    if (r < 2.5) and (nb < nbmax):
                        nb += 1
                        alm1[i] = 0
                    
                    dw = alm1[i] * d * 2 * cost[c]
                    g = s = 0
                    for W in range(10):
                        """g = 0.95 * g + (1 - 0.95) * dw * dw
                        s -= dw / (math.sqrt(g + 0.0000001)) * alm1[i]"""
                    outputw[o][i] += LR* dw 
    
				

def Loss_func(Loss2, size, layer, size_l):
	
	Loss3 = []
	err = 0
	for e in range(size):
		err += Loss2[e]
	err = err / size
	for o in range(size_l):
		Loss3.append(  err )#layer[o] *


	return Loss3


def BN(inp, size_in, scale):

	s = 0
	for i in range(size_in):
		s += inp[i]

	s /= size_in

	sx = 0
	d = 0
	for i in range(size_in):
		d += (inp[i] - s)*(inp[i] - s)
		
	d /= size_in
	
	Y = []
	for i in range(size_in):
		Y.append( ((inp[i] - s) / math.sqrt(d + 0.000001)) * scale )
		
		
	return Y
		
		

def Dropout_input(inputn):
    
    nb = 0
    nbmax = 0
    if(size_input < 4):
        nbmax = 1.0
    else:
        nbmax = size_input * 0.25
    
    print(len(inputn))
    for i in range(size_input):
        r = random.random()
        if (r <= 0.25) and (nb < nbmax):
            nb += 1
            inputn[i] = 0
            










