import random

NUM = []
NUM2c=[]
incN = 0
incN2 = 0

MAX_I = 40
MAX_IM_C = 20
MAX_RAND = 490

#testd = "archive/dog vs cat/dataset/test_set/dogs/dog."
#testc = "archive/dog vs cat/dataset/test_set/cats/cat."
#trainc ="archive/dog vs cat/dataset/training_set/cats/cat." 
#traind ="archive/dog vs cat/dataset/training_set/dogs/dog." 

testc = "archive/test/chihuahua/chihuahua."
testm = "archive/test/muffin/muffin."
trainc ="archive/test/chihuahuasel/chihuahua." 
trainm ="archive/test/muffinsel/muffin." 
trains ="archive/flowers/flower_photos/sunflowers/sun."
trainrg ="Me/image/copy/regis/regis."

testcr = "Test/croix"
testce = "Test/cercle"

PATH = [testc, testm]
PN = []

def IS_IN(tab, num):
    
    for e in tab:
        if e == num:
            return True
        
    return False
    
def load_NUM(size):
    nu = []
    for i in range(size):
        n = int(random.random() * MAX_RAND )
        while IS_IN(nu, n):
            n = int(random.random() * MAX_RAND )
        nu.append(n)
    return nu

def load_img_square():
    pn = []
    path = []

    path.append("1.jpg")
    pn.append(1)
    path.append("2.jpg")
    pn.append(2)

    return path, pn

def load_img_batch(numimg):

    NUM = load_NUM(numimg)
    NUM2 = load_NUM(numimg)  
    incN = incN2 = 0
    pn1 = []
    path1 = []
    pn2 = []
    path2 = []
    for i in range(numimg):
        path1.append(testc + str(NUM[incN]) + ".jpg")
        pn1.append(1)
        incN+=1

    for i in range(numimg):
        path2.append(trainrg + str(NUM2[incN2]) + ".jpg")
        pn2.append(2)
        incN2+=1

    return path1, pn1, path2 ,pn2

def load_img_batch_CrC(numimg):

    NUM = list(range(1,6))
    NUM2 = list(range(1,6))
    incN = incN2 = 0
    pn1 = []
    path1 = []
    pn2 = []
    path2 = []
    for i in range(numimg):
        path1.append(testcr + str(NUM[incN]) + ".png")
        pn1.append(1)
        incN+=1

    for i in range(numimg):
        path2.append(testce + str(NUM2[incN2]) + ".png")
        pn2.append(2)
        incN2+=1

    return path1, pn1, path2 ,pn2

def load_img_batch_sun(numimg):

    NUM = load_NUM(numimg)
    NUM2 = load_NUM(numimg)  
    incN = incN2 = 0
    pn1 = []
    path1 = []
    pn2 = []
    path2 = []
    for i in range(numimg):
        path1.append(testc + str(NUM[incN]) + ".jpg")
        pn1.append(1)
        incN+=1

    for i in range(numimg):
        path2.append(trainrg + str(NUM2[incN2]) + ".jpg")
        pn2.append(2)
        incN2+=1

    return path1, pn1, path2 ,pn2

def load_img_batch_3(numimg):

    NUM = load_NUM(numimg)
    NUM2 = load_NUM(numimg)  
    NUM3 = load_NUM(numimg)  

    incN = incN2 = incN3 =0
    pn1 = []
    path1 = []
    pn2 = []
    path2 = []
    pn3 = []
    path3 = []
    for i in range(numimg):
        path1.append(testc + str(NUM[incN]) + ".jpg")
        pn1.append(1)
        incN+=1

    for i in range(numimg):
        path2.append(trainrg + str(NUM2[incN2]) + ".jpg")
        pn2.append(2)
        incN2+=1

    for i in range(numimg):
        path3.append(testm + str(NUM2[incN3]) + ".jpg")
        pn3.append(3)
        incN3+=1

    return path1, pn1, path2 ,pn2, path3 ,pn3

def load_img_batch50(numimg):

    incN = incN2 = 0
    pn1 = []
    path1 = []
    pn2 = []
    path2 = []
    for i in range(numimg):
        path1.append(trainc + str(incN) + ".jpg")
        pn1.append(1)
        incN+=1

    for i in range(numimg):
        path2.append(trainrg + str(incN2) + ".jpg")
        pn2.append(2)
        incN2+=1

    return path1, pn1, path2 ,pn2

    
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
                path.append(testm + str(NUM2[incN2]) + ".jpg")
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
                path.append(trainm + str(NUM2[incN2]) + ".jpg")
                pn.append(2)
                incN2 += 1
                
    return path, pn

"""
PATH, PN = load_mix_image()


print(len(PATH))

for i in range(MAX_I):
    print(PATH[i] + " " + str(PN[i]))
    
"""


    
    
    
    
