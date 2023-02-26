from nnp import *
from im import *
from save import *
import time

display_weight_bias2();

start_time = time.time()

I = 0
I2 = 0
W = 0
WMAX = 1
stop = False
change_in = False
NORM = 0.1
valid = 0
valid_tot = np.zeros(200)
score = 0
pct = 0

lotsz = 10
alt = True


ilot = 0
while not stop:
    print("num = " + str(I))

    #if change_in:
    if alt:
        test2inputn(TEST[I], inputn)
    else:
        test2inputn(TEST2[I2], inputn)

    
    for i in range(size_input):
        #inputn[i] /= 255.0
        inputn[i] = ((inputn[i])*1000+1) / 10000.0;
    


    BN(inputn, size_input, NORM, inputn)
    #display_input(inputn)
        #Dropout_input(inputn)
        #change_in = False
        #display_test(inputn, 2)
       # print(inputn[0])

    calc_output_RELU(inputn, hiddenw[0], hiddenb[0], size_input, size_hidden, hidden[0])
    #calc_output_RELU(hidden[0], hiddenw[1], hiddenb[1], size_hidden, size_hidden2, hidden[1])
    #calc_output_RELU(hidden[1], hiddenw[2], hiddenb[2], size_hidden2, size_hidden3, hidden[2])
    #calc_output_RELU(hidden[2], hiddenw[3], hiddenb[3], size_hidden3, size_hiddenf, hidden[3])
    calc_output_RELUF(hidden[0], outputw[0], outputb[0], size_hidden, size_output_bias, output[0])

    
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
    

    print(str(I) + " : end relu")
    
    #print("------------------------------------------------------------------")
    #print(W)
   
    cost = [0, 0]
    l = 0
    #for o in range(size_output):
    #print("testout " + str(TEST_out[I]));
    if alt:
        cost[0] = (2 * (output[0][0] - 1000.0))           
        cost[1] = (2 * (output[0][1] - 0.0))   
          
    else:
        cost[0] = (2 * (output[0][0] - 0.0))           
        cost[1] = (2 * (output[0][1] - 1000.0))   
    
    #if TEST_out[I] == 1:
    #    cost[0] = (2 * (output[0][0] - 2.0))
    #else:
    #    cost[0] = (2 * (output[0][0] - 0.0))

              
    #Copy(hidden, hidden2, hiddenw, hiddenw2, hiddenb, hiddenb2, output, output2, outputw, outputw2, outputb, outputb2)
    
    if alt:
        I += 1
    else:
        I2+=1

    ilot += 1
    if ilot == lotsz:
        backprop = 10
        if backprop == 1:
            hidden_backp41(hidden[3], cost, output[0], outputw[0], outputb[0], size_output, size_output, size_hiddenf)
            hidden_backp4(hidden[2], cost, hidden[3], hiddenw[3], hiddenb[3], size_output, size_hiddenf, size_hidden3)
            hidden_backp4(hidden[1], cost, hidden[2], hiddenw[2], hiddenb[2], size_output, size_hidden3, size_hidden2)
            hidden_backp4(hidden[0], cost, hidden[1], hiddenw[1], hiddenb[1], size_output, size_hidden2, size_hidden)
            hidden_backp4(inputn, cost, hidden[0], hiddenw[0], hiddenb[0], size_output, size_hidden, size_input)
        elif backprop == 2:
            hidden_backp71(hidden[3], cost, output[0], outputw[0], outputb[0], size_output, size_output, size_hiddenf)
            hidden_backp72(hidden[2], cost, hidden[3], outputw[0], size_output, hiddenw[3], hiddenb[3], size_output, size_hiddenf, size_hidden3)
            hidden_backp72(hidden[1], cost, hidden[2], hiddenw[3], size_hiddenf, hiddenw[2], hiddenb[2], size_output, size_hidden3, size_hidden2)
            hidden_backp72(hidden[0], cost, hidden[1], hiddenw[2], size_hidden3, hiddenw[1], hiddenb[1], size_output, size_hidden2, size_hidden)
            hidden_backp72(inputn, cost, hidden[0], hiddenw[1], size_hidden2, hiddenw[0], hiddenb[0], size_output, size_hidden, size_input)
        elif backprop == 3:
            hidden_backp71(hidden[3], cost, output[0], outputw[0], outputb[0], size_output, size_output, size_hiddenf)
            hidden_backp72(hidden[2], cost, hidden[3], outputw[0], size_output, hiddenw[3], hiddenb[3], size_output, size_hiddenf, size_hidden3)
            hidden_backp73(hidden[1], cost, hidden[2], hiddenw[3], size_hiddenf, outputw[0], size_output, hiddenw[2], hiddenb[2], size_output, size_hidden3, size_hidden2)
            hidden_backp73(hidden[0], cost, hidden[1], hiddenw[2], size_hidden3, hiddenw[3], size_hiddenf, hiddenw[1], hiddenb[1], size_output, size_hidden2, size_hidden)
            hidden_backp73(inputn,    cost, hidden[0], hiddenw[1], size_hidden2, hiddenw[2], size_hidden3, hiddenw[0], hiddenb[0], size_output, size_hidden, size_input)
        elif backprop == 4:
            backprop1(cost)
            backprop2(cost, W)
            backprop3(cost, W)
            backprop4(cost, W)
            backprop5(cost, W)
        elif backprop == 5:
            backprop12(cost)
            backprop22(cost, W)
            backprop32(cost, W)
            backprop42(cost, W)
            backprop52(cost, W)
        elif backprop == 6:
            Backpropagation(0, cost[0])
            Backpropagation(1, cost[1])
        elif backprop == 7:
            if TEST_out[I] == 1:
                Backpropagation2(0, cost[0])
            else:
                Backpropagation2(1, cost[1])
        elif backprop == 8:
            Backpropagation(0, cost[0])
        elif backprop == 9:
            BackpropagationRM(cost)
        elif backprop == 10:
            Backpropagation512(cost)

        ilot = 0
        alt = not alt

    #print(hidden)
    
    
    change_in = True
    

    cnt = 0	
    E = 0		

    if I == len(TEST) and I2 == len(TEST2):
        print("epoch: " + str(W)) 
        valid = np.zeros(100)
        score, pct = Validation(TESTV, TESTV_out, NORM, valid, PATHV, TESTV2, TESTV_out2, PATHV2)
        for v in range(100):
            if valid[v]:
                valid_tot[v] = 1
        
        stop = True
        I = 0
        #W+=1

        #if W == WMAX:
        #    stop = True
            

display_weight_bias2();

PATHF = []
ilot = 0
alt = True
I=I2=0
for i in range(100):
    if alt:
        PATHF.append(PATHV[I])
        I+=1
    else:
        PATHF.append(PATHV2[I2])
        I2+=1
    ilot+=1
    if ilot == lotsz:
        ilot = 0
        alt = not alt


Afficher_images(PATHF, valid_tot, 100, score, pct)

ct = 0
for v in range(100):
    if valid_tot[v]:
        ct += 1
    
pct = ct / 100.0 * 100.0
print("---------------------------====-------------------------------")
print("Totale :" + str(pct) + "% image correctement classe")

duration = time.time() - start_time 
print("duree : " + str(duration) + " s")


               
