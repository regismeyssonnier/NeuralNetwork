from nn import *
from im import *
from save import *


#display_TEST()


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
"""
#TEST, TEST_out = create_test_tab()
#TESTV = create_valid_tab()


W = 0
WMAX = 1
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
epoch = 5
E = 0
EP = 0
INC = False
warg = 1.0
THR = 0
THRP = 0
JM = JB = 0
l = 0
lp = 0

while not stop:
	

	
	if change_in:
		inputn = test2inputn(TEST[I])
		inputn = BN(inputn, size_input, 0.001)
		Dropout_input(inputn)
		change_in = False
	#print("l:"+str(len(inputn)))
	#display_inputn(inputn)
		display_test(inputn, TEST_out[I])
		
	hidden = []
	h = []
	h = calc_output_RELU2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden)
	#print("output in hid1")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_RELU2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_RELU2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	#print("/////////////////////////////////////////////////////////////////")
	output = []
	h = []
	h = calc_output_RELUF(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
	#print("output hid2 out")
	#print(h)
	output.append(h);

	name = ""
	if (TEST_out[I]) == 1:
		name = "cat"
	elif(TEST_out[I]) == 2:
		name="chien"
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
	avg = 0.0
	for o in range(size_output):
		print("o:" + str(output[0][o]))
		avg += output[0][o]
		if output[0][o] > maxi:
			reso = o+1
			maxi = output[0][o]
	#avg /= float(size_output)
	print("avg:" + str(avg))
	"""if output[0][o] > 0.5:
		reso = 1
		maxi = output[0][o]
	else:
		reso = 2
		maxi = output[0][o]"""

	#print("---------------------------------------------------")
	"""if output[0][1] > output[0][0]:
		AVG = output[0][1] - output[0][0]
	else:
		AVG = output[0][0] - output[0][1]
	print("AVG:" + str(AVG) + " epoch:" + str(E) + " I:" + str(I+1))"""

	"""if TEST_out[I] == -1:
		print("output hid2 out")
		print(h)
		print("C'est un ZOB")
		cnt = 0
		if reso == 1:
			TEST_out[I] = 2
		elif reso == 2:
			TEST_out[I] = 1
	"""
	"""if TEST_out[I] == reso : 
	#if (output[0][0] <= (output[0][1] + 0.01)) and (output[0][0] >= (output[0][1] - 0.01)):
		print("output hid2 out")
		print(h)
		print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
		if reso == 1:
			print("C'est une cat")
		elif reso == 2:
			print("C'est un chienne")
		elif reso == 3:
			print("C'est un cercle")

		cnt = 0"""

	print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
	

	#for xx in range(epoch):
	#----------------------------------------------------------------------
	#output[0] = BN(output[0], size_output, 0.0001)
	outp = 0
	Loss = []
	cost = []
	LRL = []
	#y = []
	THRP = THR
	lp = l
	JM = JB = 0
	for o in range(size_output):
		if o == (TEST_out[I]-1):
			#Loss.append( output[0][o] * math.log(0.9) + (1-output[0][o])*math.log(1-0.9)  )
			#Loss.append( (0.33 - output[0][o])*2  )
			Loss.append( (1.0 - output[0][o])  )
			LRL.append(0.9)
			cost.append(output[0][o] - 1.0)
			#Loss.append(-math.exp(1.0) / output[0][o])
			"""if avg != 0.0:
				Loss.append( (1.0 - output[0][o]/avg)*5.0  )
				print("++" + str((1.0 - output[0][o]/avg)*5.0))
			else:
				Loss.append( 1.0  )"""
			print("++" + str( (1.0 - output[0][o]) ))
			#print("++" + str( (1.0 - output[0][o])*10.0 ))
			l += ( ((1 - output[0][o])*(1 - output[0][o])) )
			THR = 1.0 - output[0][o]
		
			JB += 1.0 - output[0][o]
			outp = output[0][o]
			#y.append(0.99)
		else:
			l += ( ((0 - output[0][o])*(0 - output[0][o])) )
			Loss.append( - output[0][o] )  # * -output[0][o])/2 )
			LRL.append(0.9)
			cost.append(output[0][o])
			#Loss.append(-math.exp(0.0) / output[0][o])
			"""if avg != 0.0:
				Loss.append( ( - (output[0][o]-0.1)/avg)  )
				print("--" + str( - (output[0][o]-0.1)/avg))
			else:
				Loss.append( 0.0  )"""
			print("--" + str( - output[0][o]))
			JB +=  - output[0][o]
			#Loss.append( output[0][o] * math.log(0.1) + (1-output[0][o])*math.log(1-0.1)  )
			#y.append(0.01)
		
		"""if output[0][o] > 0.5:
			Loss.append( (1.0 - output[0][o])  )
		else:
			Loss.append( (0.0 - output[0][o]) )"""
	
	l /= size_output
	
	for i in range(size_hidden3):
		JM += hidden[2][i] 
	
	JM *= JB
	JM *= (2 / size_output)
	JB *= (2 / size_output)
	
	#print("ERROR = " + str(l) + " I:" + str(I+1) + " E:" + str(E))
	#print("THR:" + str(THR))
	#print("---------------------------------------------------")
	"""for o in range(size_output):
		if o == (TEST_out[I]-1):
			Loss.append(l)
		else:
			Loss.append(0)"""

	#print("Loss=" + str(Loss))
	#print("Test OUT:" + str(TEST_out))

	"""
	#Hid3 Out
	hidden_backp(output[0], Loss, outputb[0], outputw[0], hidden[2], size_output, size_hidden3, JM, JB, True)

	#Hid2 Hid3
	Loss23 = Loss_func(Loss, size_output, hidden[2], size_hidden3)
	hidden_backp(hidden[2], Loss23, hiddenb[2], hiddenw[2], hidden[1], size_hidden3, size_hidden2, JM, JB, False)


	#Hid1 Hid2
	Loss2 = Loss_func(Loss23, size_output, hidden[1], size_hidden2)
	hidden_backp(hidden[1], Loss2, hiddenb[1], hiddenw[1], hidden[0], size_hidden2, size_hidden, JM, JB, False)


	#In Hid1
	Loss3 = Loss_func(Loss2, size_hidden2, hidden[0], size_hidden)
	hidden_backp(hidden[0], Loss3, hiddenb[0], hiddenw[0], inputn, size_hidden, size_input, JM, JB, False)
	"""

	#def hidden_backp3(output, Loss, outputb, outputw, layer, size_out, size_in):
	#Hid3 Out
	"""hidden_backp3(output[0], Loss, LRL, outputb[0], outputw[0], hidden[2], size_output, size_output, size_hidden3)
	hidden_backp3(hidden[2], Loss, LRL, hiddenb[2], hiddenw[2], hidden[1], size_output, size_hidden3, size_hidden2)
	hidden_backp3(hidden[1], Loss, LRL, hiddenb[1], hiddenw[1], hidden[0], size_output, size_hidden2, size_hidden)
	hidden_backp3(hidden[0], Loss, LRL, hiddenb[0], hiddenw[0], inputn, size_output, size_hidden, size_input)"""
	
	
	#hidden_backp4(alm1, cost, output, outputw, outputb, size_out, size_in):
	
        hidden_backp4(hidden[2], cost, output[0], outputw[0], outputb[0], size_output, size_output, size_hidden3)
        hidden_backp4(hidden[1], cost, hidden[2], hiddenw[2], hiddenb[2], size_output, size_hidden3, size_hidden2)
        hidden_backp4(hidden[0], cost, hidden[1], hiddenw[1], hiddenb[1], size_output, size_hidden2, size_hidden)
        hidden_backp4(inputn, cost, hidden[0], hiddenw[0], hiddenb[0], size_output, size_hidden, size_input)

	"""
	hidden_backp2(output[0], Loss, outputb[0], outputw[0], output[0], size_output, size_hidden3)

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



	cnt += 1
	E += 1
	#if E == epoch:
	#if (l < 0.01) or (AVG > 0.01):
        #if (TEST_out[I] == reso)  or (E > 20):#(outp <= 0.34)and (outp >= 0.32):#:#(output[0][TEST_out[I]-1] >= 0.33):#or ((l == lp)and(THR==THRP)):#(TEST_out[I] == reso) :#and (E > 20) :# or (TEST_out[I] == -1): 
	#if (output[0][0] <= (output[0][1] + 0.01)) and (output[0][0] >= (output[0][1] - 0.01)):
	#if l <= 0.10:
    

        I += 1
        change_in = True


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

score = 0.0
score2 = 0.0
score3 = 0
score4 = 0
nbcat = 20.0
nbdog = 20.0

W = 0
WMAX = 5
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
		inputn = test2inputn(TESTV[I])
		inputn = BN(inputn, size_input, 0.001)
		Dropout_input(inputn)
		change_in = False
	#print("l:"+str(len(inputn)))
	#display_inputn(inputn)
	#display_test(inputn, 3)
		
	hidden = []
	h = []
	h = calc_output_RELU2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden)
	#print("output in hid1")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_RELU2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	h = []
	h = calc_output_RELU2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3)
	#print("output hid1 hid2")
	#print(h)
	hidden.append(h);

	output = []
	h = []
	h = calc_output_RELUF(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
	#print("output hid2 out")
	#print(h)
	output.append(h);

		
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
		name="chien"

	

	R[resom-1]+= 1
	"""if resom == 1:
		print("cat")
	elif reso == 2:
		print("chienne")"""		 	
	

	cnt = 0

	W += 1
	if W == WMAX:

		if (TESTV_out[I]) == 1:
			mini = 10000
			reso2 = -1
			for o in range(size_output):
				if R[o] < mini:
					reso2 = o+1
					mini = R[o]

			maxi = -1
			reso = -1
			for o in range(size_output):
				if R[o] > maxi:
					reso = o+1
					maxi = R[o]

		elif (TESTV_out[I]) == 2:
			maxi = -1
			reso3 = -1
			for o in range(size_output):
				if R[o] > maxi:
					reso3 = o+1
					maxi = R[o]
					
			mini = 10000
			reso4 = -1
			for o in range(size_output):
				if R[o] < mini:
					reso4 = o+1
					mini = R[o]

		print("output hid2 out")
		print(h)
		print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))
		if reso == 1:
			if TESTV_out[I] == 1:
			#if I < 20:
				print("C'est une cat")
				score+=1.0
				score4 += 1.0
				
		elif reso == 2:
			if TESTV_out[I] == 2:
			#if I >= 20:
				print("C'est un chienne")
				score+=1.0
				score4 += 1.0
		if reso2 == 1:
			if TESTV_out[I] == 1:
			#if I < 20:
				print("2C'est une cat")
				score2+=1.0
				score3 += 1.0
		elif reso2 == 2:
			if TESTV_out[I] == 2:
			#if I >= 20:
				print("2C'est un chienne")
				score2+=1.0
				score3 += 1.0
		if reso3 == 1:
			if TESTV_out[I] == 1:
			#if I < 20:
				print("3C'est une cat")
				score+=1.0
				score2+=1.0

		elif reso3 == 2:
			if TESTV_out[I] == 2:
			#if I >= 20:
				print("3C'est un chienne")
				score+=1.0
				score2+=1.0
				
		if reso4 == 1:
			if TESTV_out[I] == 1:
			#if I < 20:
				print("4C'est une cat")
				score3 += 1.0
				score4 += 1.0
		elif reso4 == 2:
			if TESTV_out[I] == 2:
			#if I >= 20:
				print("4C'est un chienne")
				score3 += 1.0
				score4 += 1.0
				
		reso = reso2 = reso3 = reso4 = -1

		W = 0
		I += 1
		change_in = True
		R = []
		for o in range(size_output):
			R.append(0)

		if I == len(TESTV):
			stop = True


pct = float(score/(nbcat+nbdog))*100.0
print("Algo MAXMAX Score:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
pct2 = float(score2/(nbcat+nbdog))*100.0
print("Algo MINMAX Score2:" + str(score2) + "/" + str(nbcat+nbdog) + " soit " + str(pct2) + "%")
pct3 = float(score3/(nbcat+nbdog))*100.0
print("Algo MINMIN Score3:" + str(score3) + "/" + str(nbcat+nbdog) + " soit " + str(pct3) + "%")
pct4 = float(score4/(nbcat+nbdog))*100.0
print("Algo MAXMIN Score4:" + str(score4) + "/" + str(nbcat+nbdog) + " soit " + str(pct4) + "%")
print("------------------------------------------------")

sc = [score, score2, score3, score4]
maxi = -1
resoF1 = -1
for i in range(len(sc)):
	if sc[i] > maxi:
		resoF1 = i+1
		maxi = sc[i]

print("The best algo was :")
if resoF1 == 1:
	print("Algo MAXMAX = Resultat:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
elif resoF1 == 2:
	print("Algo MINMAX = Resultat:" + str(score2) + "/" + str(nbcat+nbdog) + " soit " + str(pct2) + "%")
elif resoF1 == 3:	
	print("Algo MINMIN = Resultat:" + str(score3) + "/" + str(nbcat+nbdog) + " soit " + str(pct3) + "%")
elif resoF1 == 4:	
	print("Algo MAXMIN = Resultat:" + str(score4) + "/" + str(nbcat+nbdog) + " soit " + str(pct4) + "%")
	
	
	
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")
print("--------------------------TEST-----------------------------")



score = 0.0
score2 = 0.0
score3 = 0
score4 = 0
nbcat = 20.0
nbdog = 20.0

W = 0
WMAX = 5
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
        inputn = test2inputn(TESTV[I])
        inputn = BN(inputn, size_input, 0.001)
        change_in = False
    #print("l:"+str(len(inputn)))
    #display_inputn(inputn)
    #display_test(inputn, 3)
        
    hidden = []
    h = []
    h = calc_output_RELU3(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden)
    #print("output in hid1")
    #print(h)
    hidden.append(h);

    h = []
    h = calc_output_RELU3(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2)
    #print("output hid1 hid2")
    #print(h)
    hidden.append(h);

    h = []
    h = calc_output_RELU3(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3)
    #print("output hid1 hid2")
    #print(h)
    hidden.append(h);

    output = []
    h = []
    h = calc_output_RELUF(hidden[2], outputw[0], outputb[0], size_hidden3, size_output_bias)
    #print("output hid2 out")
    #print(h)
    output.append(h);

        
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
        name="chien"

    

    R[resom-1]+= 1
    """if resom == 1:
        print("cat")
    elif reso == 2:
        print("chienne")"""		 	
    

    cnt = 0

    W += 1
    if W == WMAX:

        if (TESTV_out[I]) == 1:
            mini = 10000
            reso2 = -1
            for o in range(size_output):
                if R[o] < mini:
                    reso2 = o+1
                    mini = R[o]

            maxi = -1
            reso = -1
            for o in range(size_output):
                if R[o] > maxi:
                    reso = o+1
                    maxi = R[o]

        elif (TESTV_out[I]) == 2:
            maxi = -1
            reso3 = -1
            for o in range(size_output):
                if R[o] > maxi:
                    reso3 = o+1
                    maxi = R[o]
                    
            mini = 10000
            reso4 = -1
            for o in range(size_output):
                if R[o] < mini:
                    reso4 = o+1
                    mini = R[o]

        """print("output hid2 out")
        print(h)
        print(str(W) + " - " + str(I+1) + "=test_out:" + name + " - " + str(cnt+1))"""
        if reso == 1:
            if TESTV_out[I] == 1:
            #if I < 20:
                #print("C'est une cat")
                score+=1.0
                score4 += 1.0
                
        elif reso == 2:
            if TESTV_out[I] == 2:
            #if I >= 20:
                #print("C'est un chienne")
                score+=1.0
                score4 += 1.0
        if reso2 == 1:
            if TESTV_out[I] == 1:
            #if I < 20:
                #print("2C'est une cat")
                score2+=1.0
                score3 += 1.0
        elif reso2 == 2:
            if TESTV_out[I] == 2:
            #if I >= 20:
                #print("2C'est un chienne")
                score2+=1.0
                score3 += 1.0
        if reso3 == 1:
            if TESTV_out[I] == 1:
            #if I < 20:
                #print("3C'est une cat")
                score+=1.0
                score2+=1.0

        elif reso3 == 2:
            if TESTV_out[I] == 2:
            #if I >= 20:
                #print("3C'est un chienne")
                score+=1.0
                score2+=1.0
                
        if reso4 == 1:
            if TESTV_out[I] == 1:
            #if I < 20:
                #print("4C'est une cat")
                score3 += 1.0
                score4 += 1.0
        elif reso4 == 2:
            if TESTV_out[I] == 2:
            #if I >= 20:
                #print("4C'est un chienne")
                score3 += 1.0
                score4 += 1.0
                
        reso = reso2 = reso3 = reso4 = -1

        W = 0
        I += 1
        change_in = True
        R = []
        for o in range(size_output):
            R.append(0)

        if I == len(TESTV):
            stop = True


pct = float(score/(nbcat+nbdog))*100.0
print("Algo MAXMAX Score:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
pct2 = float(score2/(nbcat+nbdog))*100.0
print("Algo MINMAX Score2:" + str(score2) + "/" + str(nbcat+nbdog) + " soit " + str(pct2) + "%")
pct3 = float(score3/(nbcat+nbdog))*100.0
print("Algo MINMIN Score3:" + str(score3) + "/" + str(nbcat+nbdog) + " soit " + str(pct3) + "%")
pct4 = float(score4/(nbcat+nbdog))*100.0
print("Algo MAXMIN Score4:" + str(score4) + "/" + str(nbcat+nbdog) + " soit " + str(pct4) + "%")
print("------------------------------------------------")

sc = [score, score2, score3, score4]
maxi = -1
resoF2 = -1
for i in range(len(sc)):
    if sc[i] > maxi:
        resoF2 = i+1
        maxi = sc[i]

print("The best algo was without dropout:")
if resoF2 == 1:
    print("Algo MAXMAX = Resultat:" + str(score) + "/" + str(nbcat+nbdog) + " soit " + str(pct) + "%")
elif resoF2 == 2:
    print("Algo MINMAX = Resultat:" + str(score2) + "/" + str(nbcat+nbdog) + " soit " + str(pct2) + "%")
elif resoF2 == 3:	
    print("Algo MINMIN = Resultat:" + str(score3) + "/" + str(nbcat+nbdog) + " soit " + str(pct3) + "%")
elif resoF2 == 4:	
    print("Algo MAXMIN = Resultat:" + str(score4) + "/" + str(nbcat+nbdog) + " soit " + str(pct4) + "%")
