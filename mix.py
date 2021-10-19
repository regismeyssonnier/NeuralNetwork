import random

NUM = []
NUM2c=[]
incN = 0
incN2 = 0

MAX_I = 40
MAX_IM_C = 20
MAX_RAND = 100

testd = "archive/dog vs cat/dataset/test_set/dogs/dog"
testc = "archive/dog vs cat/dataset/test_set/cats/cat"
trainc ="archive/dog vs cat/dataset/training_set/cats/cat" 
traind ="archive/dog vs cat/dataset/training_set/dogs/dog." 
PATH = [testc, testd]
PN = []

def IS_IN(tab, num):
    
    for e in tab:
        if e == num:
            return True
        
    return False
    
def load_NUM(size):
    nu = []
    for i in range(size):
        n = int(random.random() * MAX_RAND + 1)
        while IS_IN(nu, n):
            n = int(random.random() * MAX_I + 1)
        nu.append(n)
    return nu
    
def load_mix_image_valid():
    
        
    incN = incN2 = 0
    
    NUM = load_NUM(MAX_IM_C)
    NUM2 = load_NUM(MAX_IM_C)   
    
    """print(NUM)
    print(len(NUM))
    print(NUM2)
    print(len(NUM2))"""
    
    pn = []
    path = []
    while (incN < 20) or (incN2 < 20) :
        n = random.random()
        if n <= 0.5:
            if incN < MAX_IM_C:
                path.append(testc + str(NUM[incN]) + ".jpg")
                pn.append(1)
                incN += 1
        else:
            if incN2 < MAX_IM_C:
                path.append(testd + str(NUM2[incN2]) + ".jpg")
                pn.append(2)
                incN2 += 1
                
    return path, pn


def load_mix_image_train():
    
        
    incN = incN2 = 0
    
    NUM = load_NUM(MAX_IM_C)
    NUM2 = load_NUM(MAX_IM_C)   
    
    """print(NUM)
    print(len(NUM))
    print(NUM2)
    print(len(NUM2))"""
    
    pn = []
    path = []
    while (incN < 20) or (incN2 < 20) :
        n = random.random()
        if n <= 0.5:
            if incN < MAX_IM_C:
                path.append(trainc + str(NUM[incN]) + ".jpg")
                pn.append(1)
                incN += 1
        else:
            if incN2 < MAX_IM_C:
                path.append(traind + str(NUM2[incN2]) + ".jpg")
                pn.append(2)
                incN2 += 1
                
    return path, pn

"""
PATH, PN = load_mix_image()


print(len(PATH))

for i in range(MAX_I):
    print(PATH[i] + " " + str(PN[i]))
    
"""


    
    
    
    
