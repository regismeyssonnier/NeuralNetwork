#coding=utf-8
from gtts import gTTS as sp
import os


class MySpeech:

    def __init__(self, text, lang):
        self.text = text
        self.lang = lang


    def speak(self):

        speech = sp(text=self.text, lang=self.lang, slow=False)
        speech.save("./temp/speech.mp3")
        os.popen("start temp/speech.mp3")


#text = 'Bonjour, Regis tu es le plus fort. Comment vas tu'
#myspeech = MySpeech(text, "fr")
#myspeech.speak()