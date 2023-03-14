#coding:utf-8
import spacy
from myspeech import *


class MyRecognition:

    def __init__(self):
        self.nlp = spacy.load('fr_core_news_sm')


    def set_text(self, text):
        self.text = text.lower()
        print(self.text)

    def analyze(self):
        if self.text == "":
            print("set the text before")

        doc = self.nlp(self.text)
        print(doc)

        self.commande = False
        self.jouer = False

        self.filtre = False
        self.sobel = False
        self.filtre_sobel_noir = True
        self.couleur = False
        self.couleur_rouge = False
        self.couleur_vert = False
        self.couleur_bleu = False

        for token in doc:
            print(token.lemma_ + " " + token.pos_)
            if token.lemma_ ==  "mettre" or token.lemma_ == "mets" or token.lemma_ == "mais":
                print("commande")
                self.commande = True
                
            
            if token.lemma_ ==  "jouer" or token.lemma_ ==  "d\xe9buter" or \
               token.lemma_ ==  "commencer":
                print("jouer")
                self.jouer = True
                        
            if "jeu" in token.text:
                self.jouer = True
            if "filtre" in token.lemma_:
                self.filtre = True
            if "s" == token.lemma_:
                self.sobel = True
                self.filtre_sobel_noir = True
            if "blanc" in token.text:
                self.filtre_sobel_noir = False
            if "couleur" in token.lemma_:
                self.couleur = True
            if "vert" in token.text:
                self.couleur_vert = True
            if "rouge" in token.text:
                self.couleur_rouge = True
            if "bleu" in token.text:
                self.couleur_bleu = True
            

        self.num_commande = 0
        self.answer = ""
        if self.commande:
            self.answer += "Je met "

            if self.jouer:
                self.answer += "le jeu "
                self.num_commande = "jeu"

            if self.filtre:
                if self.filtre:
                    self.answer += "le filtre "
                    if self.sobel:
                        self.num_commande = "sobeln"
                        if self.filtre_sobel_noir:
                            self.answer += "le filtre sobel noir "
                        else:
                            self.answer += "le filtre sobel blanc "
                            self.num_commande = "sobelbl"

                    if self.couleur:
                        if self.couleur_vert:
                            self.answer += "le filtre couleur vert "
                            self.num_commande = "colvert"
                        elif self.couleur_rouge:
                            self.answer += "le filtre couleur rouge "
                            self.num_commande = "colrouge"
                        elif self.couleur_bleu:
                            self.answer += "le filtre couleur bleu "
                            self.num_commande = "colbleu"
                        else:
                            self.answer += "le filtre couleur vert "
                            self.num_commande = "colvert"
                                       

        if self.jouer:
            self.answer = "Je vais mettre le jeu."
            self.num_commande = "jeu"
            if self.filtre:
                self.answer += "sans les filtres."

        return self.answer, self.num_commande

"""
myreco = MyRecognition()
myreco.set_text("tu mets le filtre s noir")
text, num = myreco.analyze()
myspeech = MySpeech(text, "fr")
myspeech.speak()
"""