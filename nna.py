from nn import *
from im import *
from save import *


#display_TEST()

"""
random_weight()
display_hiddenw()

random_bias("sigmoid")
display_hiddenb()

random_outputw()
display_outputw()

random_outputb()
display_outputb()
"""

hiddenw = []
load_hidden_weight(hiddenw, nb_hidden)

hiddenb = []
load_hidden_bias(hiddenb, nb_hidden)

outputw = []
load_output_weight(outputw)

outputb = []
load_output_bias(outputb)


W = 0
WMAX = 23
I = 0
cnt = 0
stop = False
X = 0
#for X in range(10000):
y1 = [1.0, 0.0, 0.0]
y2 = [0.0, 1.0, 0.0]
y3 = [0.0, 0.0, 1.0]
test = 0
Y = y1
change_in = True
epoch = 20
E = 0
EP = 0

while not stop:
	

	#del inputn[:]
	#inputn = []
	if change_in:
		inputn = test2inputn(TEST[I])
		change_in = False
	#print("l:"+str(len(inputn)))
	#display_inputn(inputn)
	#display_test(inputn, TEST_out[I])
		
	hidden = []
	h = []
	h = calc_output_sigmoid2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden)
	#print("output in hid1")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_sigmoid2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_sigmoid2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	output = []
	h = []
	h = calc_output_sigmoid2(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
	#print("output hid2 out")
	#print(h)
	output.append(h);

	name = ""
	if (TEST_out[I]) == 1:
		name = "croix"
	elif(TEST_out[I]) == 2:
		name="trait"
	elif(TEST_out[I]) == 3:
		name="cercle"

	"""if test+1 == 1:
		name = "croix"
	elif test+1 == 2:
		name="trait"
	elif test+1 == 3:
		name="cercle"""
	
	maxi = -1
	reso = -1
	for o in range(size_output):
		if output[0][o] > maxi:
			reso = o+1
			maxi = output[0][o]

	if TEST_out[I] == reso : 
	 
		print("output hid2 out")
		print(h)
		print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
		if reso == 1:
			print("C'est une croix")
		elif reso == 2:
			print("C'est un trait")
		elif reso == 3:
			print("C'est un cercle")

		cnt = 0

	#for xx in range(epoch):
	#----------------------------------------------------------------------
	Loss = []
	#y = []
	l = 0
	for o in range(size_output):
		if o == (TEST_out[I]-1):
			#Loss.append( output[0][o] * math.log(0.9) + (1-output[0][o])*math.log(1-0.9)  )
			#Loss.append( (0.33 - output[0][o])*2  )
			Loss.append( -(1 - output[0][o])  )
			l += ( ((1 - output[0][o])*(1 - output[0][o])) )
			#y.append(0.99)
		else:
			l += ( ((0 - output[0][o])*(0 - output[0][o])) )
			Loss.append( -(0 - output[0][o]) )  # * -output[0][o])/2 )
			#Loss.append( output[0][o] * math.log(0.1) + (1-output[0][o])*math.log(1-0.1)  )
			#y.append(0.01)
	
		"""if output[0][o] > 0.5:
			Loss.append( (1.0 - output[0][o])  )
		else:
			Loss.append( (0.0 - output[0][o]) )"""
	
	l /= size_output*2
	print("ERROR = " + str(l))

	"""for o in range(size_output):
		if o == (TEST_out[I]-1):
			Loss.append(l)
		else:
			Loss.append(0)"""

	#print("Loss=" + str(Loss))
	#print("Test OUT:" + str(TEST_out))


	#Hid3 Out
	hidden_backp(output[0], Loss, outputb[0], outputw[0], hidden[2], size_output, size_hidden3,l)

	#Hid2 Hid3
	Loss23 = Loss_func(Loss, size_output, hidden[2], size_hidden3)
	hidden_backp(hidden[2], Loss23, hiddenb[2], hiddenw[2], hidden[1], size_hidden3, size_hidden2,l)


	#Hid1 Hid2
	Loss2 = Loss_func(Loss23, size_output, hidden[1], size_hidden2)
	hidden_backp(hidden[1], Loss2, hiddenb[1], hiddenw[1], hidden[0], size_hidden2, size_hidden,l)


	#In Hid1
	Loss3 = Loss_func(Loss2, size_hidden2, hidden[0], size_hidden)
	hidden_backp(hidden[0], Loss3, hiddenb[0], hiddenw[0], inputn, size_hidden, size_input,l)


	"""hidden_backp2(output[0], Loss, outputb[0], outputw[0], output[0], size_output, size_hidden3)

	#Hid2 Hid3
	Loss23 = Loss_func(Loss, size_output, hidden[2], size_hidden3)
	hidden_backp2(hidden[2], Loss23, hiddenb[2], hiddenw[2], hidden[2], size_hidden3, size_hidden2)


	#Hid1 Hid2
	Loss2 = Loss_func(Loss23, size_output, hidden[1], size_hidden2)
	hidden_backp2(hidden[1], Loss2, hiddenb[1], hiddenw[1], hidden[1], size_hidden2, size_hidden)


	#In Hid1
	Loss3 = Loss_func(Loss2, size_hidden2, hidden[0], size_hidden)
	hidden_backp2(hidden[0], Loss3, hiddenb[0], hiddenw[0], hidden[0], size_hidden, size_input)
	"""
	"""W+=1
	if W == WMAX:
		I+=1
		W = 0"""

	cnt += 1
	E += 1
	#if E == epoch:
	if TEST_out[I] == reso: 
	#if l <= 0.10:
		I += 1
		change_in = True

		"""print("output hid2 out")
			print(h)
			print(str(W) + " - " + str(I) + "=test_out:" + name + " - " + str(cnt))
			if reso == 1:
				print("C'est une croix")
			elif reso == 2:
				print("C'est un trait")
			elif reso == 3:
				print("C'est un cercle")
		"""

		cnt = 0	
		E = 0		
		
	if I == len(TEST):
		I = 0
		W+=1
	
		if W == WMAX:
			stop = True

#save the NN
save_hidden_weight(nb_hidden, hiddenw)
#hiddenw = []
#load_hidden_weight(hiddenw, nb_hidden)
#display_hidden_weight(hiddenw, nb_hidden)

save_hidden_bias(nb_hidden, hiddenb)
#hiddenb = []
#load_hidden_bias(hiddenb, nb_hidden)
#display_hidden_bias(hiddenb, nb_hidden)

save_output_weight(outputw)
#outputw = []
#load_output_weight(outputw)
#display_hidden_weight(outputw, 1)

save_output_bias(outputb)
#outputb = []
#load_output_bias(outputb)
#display_hidden_bias(outputb, 1)


print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")

for I in range(len(TESTV)):

	#if change_in:
	inputn = test2inputn(TESTV[I])
	#	change_in = False
	#print("l:"+str(len(inputn)))
	#display_inputn(inputn)
	#display_test(inputn, TEST_out[I])
		
	hidden = []
	h = []
	h = calc_output_sigmoid2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden)
	#print("output in hid1")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_sigmoid2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_sigmoid2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	output = []
	h = []
	h = calc_output_sigmoid2(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
	#print("output hid2 out")
	#print(h)
	output.append(h);

		
	maxi = -1
	reso = -1
	for o in range(size_output):
		if output[0][o] > maxi:
			reso = o+1
			maxi = output[0][o]

			 
	print("output hid2 out")
	print(h)
	print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
	if reso == 1:
		print("C'est une croix")
	elif reso == 2:
		print("C'est un trait")
	elif reso == 3:
		print("C'est un cercle")

	cnt = 0

	
