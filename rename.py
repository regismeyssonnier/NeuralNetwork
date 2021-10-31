import os

trainc ="archive/dog vs cat/dataset/test_set/cats/" 
traind ="archive/dog vs cat/dataset/test_set/dogs/" 

ListeFichiers = os.listdir(trainc)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(trainc, fl), os.path.join(trainc, ''.join(["cat.", str(index), '.jpg'])))
    
ListeFichiers = os.listdir(traind)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(traind, fl), os.path.join(traind, ''.join(["dog.", str(index), '.jpg'])))
