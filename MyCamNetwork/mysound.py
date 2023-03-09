import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound
import numpy as np
class MySound:

    def __init__(self, name):
        self.name= name
        
    
    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = indata

    
        
    def record(self, sec, hz, ch):
        self.fs = hz
        self.duration = sec
        self.channels = ch
        self.myrec = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels, dtype='int16')
        #sd.wait()
        

    def save(self):
        write(self.name, self.fs, self.myrec) 

    def wait(self):
        sd.wait()

    def play(self):
        playsound(self.name)

"""mysound = MySound("regis.wav")
mysound.record(3, 16000, 1)
mysound.wait()
mysound.save()
mysound.play()"""