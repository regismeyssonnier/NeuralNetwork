

import random
import math


inputn = []
size_input = 2500
hidden = []
hiddenw = []
hiddenb = []
nb_hidden = 2
size_hidden = 16
size_hidden2 = 8
sz_hidden = [size_hidden, size_hidden2]
size_hidden_weight = [size_input, size_hidden]
size_hidden_bias = [size_hidden, size_hidden2]

output = []
outputw = []
outputb = []
size_output = 2
size_output_weight = size_hidden2
size_output_bias = size_output
LR = 0.1

TEST = []
TEST_out = []


def test_proc(test, out):
	
	TEST.append(test)
	TEST_out.append(out)
	

def display_test(test, out):

	print("output :  " + str(out))
	for i in range(5):
		d = ''
		for j in range(5):
			d += str(test[i*5+j]) + " "
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

	"""
	print("nb_hidden:" + str(nb_hidden))
	for nb in range(nb_hidden):#2
		for nh in range(len(sz_hidden)):
			print("sz_hidden:" + str(sz_hidden[nh]))
			for n in range(sz_hidden[nh]):#16
				print(n)
				w = []
				print("size[n]:" + str(size[n]))
				for i in range(size[n]):
					w.append(random.random())

				#hid.append(w)
	"""

	
	for no in range(szo):
		wl = []
		for nw in range(szw):
			wl.append(random.random())

		hid.append(wl)


	

def random_hidden_bias_RELU(hid, size):

	for n in range(nb_hidden):
		
		w = []
		for i in range(size[n]):
			ns = random.random()
			nb = random.random()*100
			if ns > 5:
				w.append(nb)
			else:
				w.append(-nb)
		hid.append(w)

def random_hidden_bias_sigmoid(hid, size):

	for n in range(nb_hidden):
		
		w = []
		for i in range(size[n]):
			w.append(random.random())

		hid.append(w)

def random_output_w(hid, szo, szw):
		
	"""w = []
	for i in range(size):
		w.append(random.random())

	hid.append(w)"""
	
	for no in range(szo):
		wl = []
		for nw in range(szw):
			wl.append(random.random())

		hid.append(wl)

def random_output_bias_sigmoid(hid, size):

	
	w = []
	for i in range(size):
		w.append(random.random()*100)

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
	random_output_w(outputw[0], 2, size_hidden2)

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
		
	return 1 / (1 + math.exp(-x))
	"""except OverflowError:
		print("x:" + str(x))
		#print("exp:" + str(math.exp(-x)))
		return 1 / (1 + math.exp(0))
	"""
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

def calc_output_sigmoid2(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]

		out.append( sigmoid( ri + hb[o] ) )

	return out

# [ [16[25]], [16[16]] ]
def calc_output_RELU2(inp, hw, hb, size_in, size_out):
	out = []
	for o in range(size_out):
		ri = 0
		for i in range(size_in):
			ri += inp[i] * hw[o][i]

		out.append( RELU( ri + hb[o] ) )

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



















