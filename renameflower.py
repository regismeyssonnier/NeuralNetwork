import os

trainc ="archive/flowers/flower_photos/sunflowers" 
traind ="archive/test/muffinsel/" 
trainc ="Me/image/copy/regis/"

ListeFichiers = os.listdir(trainc)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(trainc, fl), os.path.join(trainc, ''.join(["regis.", str(index), '.jpg'])))
    
    """
ListeFichiers = os.listdir(traind)

for index, fl in enumerate(ListeFichiers):
    os.rename(os.path.join(traind, fl), os.path.join(traind, ''.join(["muffin.", str(index), '.jpg'])))
    """