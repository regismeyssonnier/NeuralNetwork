from nn import *
from im import *



#display_TEST()

random_weight()
display_hiddenw()



random_bias("sigmoid")
display_hiddenb()


random_outputw()
display_outputw()


random_outputb()
display_outputb()


I = 0
for X in range(10000):

	

	#del inputn[:]
	inputn = []
	inputn = test2inputn(TEST[I])
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

	output = []
	h = []
	h = calc_output_sigmoid2(hidden[1], outputw[0], outputb[0], size_output_weight, size_output_bias)
	print("output hid2 out")
	print(h)
	output.append(h);

	name = ""
	if (TEST_out[I]-1) == 0:
		name = "croix"
	else:
		name="trait"

	print(str(X) + " - " + str(I) + "=test_out:" + name)
	if output[0][0] > output[0][1]:
		print("C'est une croix")
	else:
		print("C'est un trai")

	#----------------------------------------------------------------------
	Loss = []

	"""
	dEtot/dw5 = dEtot/dout1 * dout1/dnet1 * dnet1/dw5


	dEtot/dout1 = -(target1 - out1)
	dout1/dnet1 = dout1 x (1 - dout1)
	dnet1/dw5 = dout1

	w5' = w5 - lr x dEtot/dw5

	dE1/douth1 = dE1/dnet1 * dnet1/douth1
			0.1384 * 0.4(w5)

	"""
	
	for o in range(size_output):
		if o == (TEST_out[I]-1):
			Loss.append( (1 - output[0][o]) )#*(1 - output[0][o])) )
		else:
			Loss.append( -output[0][o] )#* (0-output[0][o])))
	print("Loss=" + str(Loss))
	#print("Test OUT:" + str(TEST_out))
	
	#Hid2 Out
	for o in range(size_output):
		d = derive(output[0][o])
		db = LR * Loss[o] * d
		outputb[0][o] = db
		for i in range(size_hidden2):
			outputw[0][o][i] = db * hidden[1][i]


	#Hid1 Hid2
	Loss2 = []
	err = (Loss[0]+Loss[1])*0.5
	for o in range(size_hidden2):
		Loss2.append(err)# * hidden[1][o])#Loss[0] +  hidden[1][o] * Loss[1])
		
	#print("Loss2=" + str(Loss2))
	

	#Hid1 Hid2
	for o in range(size_hidden2):
		d = derive(hidden[1][o])
		db = LR * Loss2[o] * d
		hiddenb[1][o] = db
		for i in range(size_hidden):
			hiddenw[1][o][i] = db * hidden[0][i]


	#In Hid1
	Loss3 = []
	err = 0
	for e in range(size_hidden2):
		err += Loss2[e]
	err = err / 16
	for o in range(size_hidden):
		Loss3.append( err )#* hidden[0][o] )#Loss2[i] )
		
	#print("Loss3=" + str(Loss3))


	#In Hid1
	for o in range(size_hidden):
		d = derive(hidden[0][o])
		db = LR * Loss3[o] * d
		hiddenb[0][o] = db
		for i in range(size_input):
			hiddenw[0][o][i] = db * inputn[i]

	I+=1
	if I == len(TEST):
		I = 0


