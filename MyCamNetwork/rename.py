import os

trainc ="image/copy/autre/" 
traind ="archive/test/muffinsel/" 

ListeFichiers = os.listdir(trainc)

for index, fl in enumerate(ListeFichiers):
    nom_fichier, extension = os.path.splitext(fl)
    os.rename(os.path.join(trainc, fl), os.path.join(trainc, ''.join(["autre.", str(index), extension])))
    
    """
ListeFichiers = os.listdir(traind)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(traind, fl), os.path.join(traind, ''.join(["muffin.", str(index), '.jpg'])))
    """