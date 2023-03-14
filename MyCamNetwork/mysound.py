import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound
import numpy as np
from vosk import Model, KaldiRecognizer
import ctypes
import queue



class MySound:

    def __init__(self, name):
        self.name= name
        
    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))
        
        
    def record(self, sec, hz, ch):
        self.fs = hz
        self.duration = sec
        self.channels = ch
        self.myrec = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels, dtype='int16')
        sd.wait()
        
    def recognize(self):
        self.q = queue.Queue()

        self.model = Model(model_name="vosk-model-small-fr-0.22")

        self.recon = KaldiRecognizer(self.model, 16000)
        self.recon.SetWords(True)
        t = ""
        with sd.RawInputStream(samplerate=16000, blocksize = 1024, device=1, dtype='int16',
                            channels=1, latency='high', callback=self.callback):
            print('#' * 80)
            print('Press Ctrl+C to stop the recording')
            print('#' * 80)
 
            rec = KaldiRecognizer(self.model, 16000)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    r = eval(rec.Result())
                    if r['text'] != "stop":
                        t += r["text"]
                    com = r["text"]
                    if com == "annul\xe9":
                        t = ""
                    print(t)
                    print(com)
                    if com == "stop":
                        break

        return t

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

"""
mysound = MySound("regis.wav")
#mysound.record(3, 16000, 1)
mysound.recognize()
"""