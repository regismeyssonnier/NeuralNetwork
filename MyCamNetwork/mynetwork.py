import cv2
import sys
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import matplotlib.pyplot as plt

print(tf.__version__)

import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt


class MyNetwork:

    def __init__(self, class_name, imw, imh, dataset_url):
        self.classname = class_name
        self.w = imw
        self.h = imh
        self.dataset_url = dataset_url
        self.model = 0

    def init_model(self, bsz, valid_sp):
        self.batch_size = bsz
        self.validation_rate = valid_sp

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
          pathlib.Path(self.dataset_url),
          validation_split=self.validation_rate,
          subset="training",
          seed=123,
          image_size=(self.h, self.w),
          batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
          pathlib.Path(self.dataset_url),
          validation_split=self.validation_rate,
          subset="validation",
          seed=123,
          image_size=(self.h, self.w),
          batch_size=self.batch_size)
            

        print(self.train_ds.class_names)

    def create_model(self):

        self.model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        
    def train_model(self, epochs):

        self.epochs = epochs

        self.history = self.model.fit(
          self.train_ds,
          validation_data=self.val_ds,
          epochs=self.epochs
        )

        self.model.summary()

    def get_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        epochs_range = range(self.epochs)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def summary(self):
        self.model.summary()

    def save_model(self,path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, path):
        p_url = path
        p_path = pathlib.Path(p_url)

        img = tf.keras.utils.load_img(
            p_path, target_size=(self.h, self.w)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(predictions[0])
        regis= ["pas moi", "Regis"]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(regis[int(np.argmax(score))], 100 * np.max(score))
        )

        if int(np.argmax(score)) == 1:
            print("HEY, C'EST MOI REGIS !!!!!!!!!!")
            return 1
        return -1

    def predict_mem(self, img):
        """p_url = path
        p_path = pathlib.Path(p_url)

        img = tf.keras.utils.load_img(
            p_path, target_size=(self.h, self.h)
        )"""
        #img_pil = tf.keras.utils.array_to_img(img)
        #img_array = tf.keras.utils.img_to_array(img_pil)
        img_array = tf.expand_dims(img, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(predictions[0])
        regis= ["pas moi", "Regis"]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(regis[int(np.argmax(score))], 100 * np.max(score))
        )

        if int(np.argmax(score)) == 1:
            print("HEY, C'EST MOI REGIS !!!!!!!!!!")
            return 1
        return -1



mynet = MyNetwork("regis", 180, 180, "image/copy")
"""
mynet.init_model(25, 0.2)
mynet.create_model()
mynet.train_model(10)
mynet.get_history()
mynet.save_model("model/regismodel")

"""
mynet.load_model("model/regismodel")
#mynet.summary()
mynet.predict("image/test.png")
