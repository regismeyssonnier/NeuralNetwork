from nnp import *
from im import *
from save import *
import time

start_time = time.time()

I = 0
W = 0
WMAX = 10
stop = False
change_in = False
NORM = 0.001
valid = 0
valid_tot = np.zeros(40)
score = 0
pct = 0

while not stop:

    if change_in:
        test2inputn(TEST[I], inputn)
        BN(inputn, size_input, NORM, inputn)
        Dropout_input(inputn)
        change_in = False
    
    calc_output_RELU2(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
    calc_output_RELU2(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
    calc_output_RELU2(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
    calc_output_RELU2(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
    calc_output_RELUF(hidden[3], outputw[0], outputb[0], size_hiddenf, size_output_bias, output[0])

    #print(str(I) + " : " + str(output[0]))
    
    #print("------------------------------------------------------------------")
    #print(W)
   
    cost = []
    l = 0
    for o in range(size_output):
        if o == (TEST_out[I]-1):
                        
            cost.append(output[0][o] - 1.0)
            
            l += ( ((1 - output[0][o])*(1 - output[0][o])) )
          
        else:
            l += ( ((0 - output[0][o])*(0 - output[0][o])) )
            
            cost.append(output[0][o] - 0.00)
                        

    l /= size_output
    
    hidden_backp41(hidden[3], cost, output[0], outputw[0], outputb[0], size_output, size_output, size_hiddenf)
    hidden_backp4(hidden[2], cost, hidden[3], hiddenw[3], hiddenb[3], size_output, size_hiddenf, size_hidden3)
    hidden_backp4(hidden[1], cost, hidden[2], hiddenw[2], hiddenb[2], size_output, size_hidden3, size_hidden2)
    hidden_backp4(hidden[0], cost, hidden[1], hiddenw[1], hiddenb[1], size_output, size_hidden2, size_hidden)
    hidden_backp4(inputn, cost, hidden[0], hiddenw[0], hiddenb[0], size_output, size_hidden, size_input)
    
    
    

    I += 1
    change_in = True

    cnt = 0	
    E = 0		

    if I == len(TEST):
        print("epoch: " + str(W)) 
        valid = np.zeros(40)
        score, pct = Validation(TESTV, TESTV_out, NORM, valid)
        for v in range(40):
            if valid[v]:
                valid_tot[v] = True
        
       
        I = 0
        W+=1

        if W == WMAX:
            stop = True
            


Afficher_images(PATHV, valid_tot, 40, score, pct)

ct = 0
for v in range(40):
    if valid_tot[v]:
        ct += 1
    
pct = ct / 40.0 * 100.0
print("---------------------------====-------------------------------")
print("Totale :" + str(pct) + "% image correctement classe")

duration = time.time() - start_time 
print("duree : " + str(duration) + " s")

