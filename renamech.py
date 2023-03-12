import os

trainc ="archive/test/chihuahuasel/" 
traind ="archive/test/muffinsel/" 

ListeFichiers = os.listdir(trainc)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(trainc, fl), os.path.join(trainc, ''.join(["chihuahua.", str(index), '.jpg'])))
    
ListeFichiers = os.listdir(traind)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(traind, fl), os.path.join(traind, ''.join(["muffin.", str(index), '.jpg'])))
