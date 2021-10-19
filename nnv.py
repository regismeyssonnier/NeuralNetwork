from nn import *
from imv import *
from save import *

#save the NN
#save_hidden_weight(nb_hidden, hiddenw)
hiddenw = []
load_hidden_weight_v(hiddenw, nb_hidden)
#display_hidden_weight(hiddenw, nb_hidden)

#save_hidden_bias(nb_hidden, hiddenb)
hiddenb = []
load_hidden_bias_v(hiddenb, nb_hidden)
#display_hidden_bias(hiddenb, nb_hidden)

#save_output_weight(outputw)
outputw = []
load_output_weight_v(outputw)
#display_hidden_weight(outputw, 1)

#save_output_bias(outputb)
outputb = []
load_output_bias_v(outputb)
#display_hidden_bias(outputb, 1)


print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
score = 0.0
nbcat = 20.0
nbdog = 20.0


for I in range(len(TESTV)):

	#if change_in:
	inputn = test2inputn(TESTV[I])
	inputn = BN(inputn, size_input, 0.000001)
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
	h = calc_output_sigmoid3(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
	#print("output hid2 out")
	#print(h)
	output.append(h);

	
	maxi = -1
	reso = -1
	for o in range(size_output):
		if output[0][o] > maxi:
			reso = o+1
			maxi = output[0][o]

	#if TEST_out[I] == reso: 
		 
	print("output hid2 out")
	print(h)
	#print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
	print("num: " + str(I+1))
	if reso == 1:
		print("C'est une cat")
		if I < 20:
			score+=1.0
	elif reso == 2:
		print("C'est un chien")
		if I >= 20:
			score+=1.0
	elif reso == 3:
		print("C'est un cercle")

	cnt = 0


pct = float(score/(nbcat+nbdog))*100.0
print("Score:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")


