import collections
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import TextVectorization
import pathlib
import matplotlib.pyplot as plt


class MyVocal:

    def __init__(self, bsz):
        self.batch_size = bsz
        self.seed = 42
        self.AUTOTUNE = tf.data.AUTOTUNE

        self.classes = ['commande', 'connaitre', 'faire', 'autre']        

    def init_model(self):

        self.train_dir = pathlib.Path("text/train")

        self.raw_train_ds = utils.text_dataset_from_directory(
            self.train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='training',
            seed=self.seed)
        
               
        for i, label in enumerate(self.raw_train_ds.class_names):
          print("Label", i, "corresponds to", label)

        # Create a validation set.
        raw_val_ds = utils.text_dataset_from_directory(
            self.train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation',
            seed=self.seed)

        self.test_dir = pathlib.Path("text/test")


        # Create a test set.
        self.raw_test_ds = utils.text_dataset_from_directory(
            self.test_dir,
            batch_size=self.batch_size)

        self.VOCAB_SIZE = 10000

      
        self.MAX_SEQUENCE_LENGTH = 250

        self.int_vectorize_layer = TextVectorization(
            max_tokens=self.VOCAB_SIZE,
            output_mode='int',
            output_sequence_length=self.MAX_SEQUENCE_LENGTH)

        # Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
        train_text = self.raw_train_ds.map(lambda text, labels: text)
        self.int_vectorize_layer.adapt(train_text)
  

        # Retrieve a batch (of 32 reviews and labels) from the dataset.
        text_batch, label_batch = next(iter(self.raw_train_ds))
        first_question, first_label = text_batch[0], label_batch[0]
        print("Question", first_question)
        print("Label", first_label)

        print("'int' vectorized question:",
              self.int_vectorize_text(first_question, first_label)[0])

        
        self.int_train_ds = self.raw_train_ds.map(self.int_vectorize_text)
        self.int_val_ds = raw_val_ds.map(self.int_vectorize_text)
        self.int_test_ds = self.raw_test_ds.map(self.int_vectorize_text)

        AUTOTUNE = tf.data.AUTOTUNE

        
        self.int_train_ds = self.configure_dataset(self.int_train_ds)
        self.int_val_ds = self.configure_dataset(self.int_val_ds)
        self.int_test_ds = self.configure_dataset(self.int_test_ds)


    def create_model(self):

        self.model = tf.keras.Sequential([
              layers.Embedding(self.VOCAB_SIZE, 64),
              #layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
              #layers.GlobalMaxPooling1D(),
              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
              layers.Dense(4)
          ])

        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])

        history = self.model.fit(self.int_train_ds, validation_data=self.int_val_ds, epochs=100)

        print("ConvNet model on int vectorized data:")
        print(self.model.summary())

        self.export_model = tf.keras.Sequential(
            [self.int_vectorize_layer, self.model,
             layers.Activation('sigmoid')])

        self.export_model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy'])


        # Test it with `raw_test_ds`, which yields raw strings
        loss, accuracy = self.export_model.evaluate(self.raw_test_ds)
        print("Accuracy: {:2.2%}".format(accuracy))

        self.export_model.summary()
                
        #test
        inputs = [
            "enculer",
            "salope",
            "je te connais"
        ]
        predicted_scores = self.export_model.predict(inputs)
        predicted_labels = self.get_string_labels(predicted_scores)
        for input, label in zip(inputs, predicted_labels):
          print("Question: ", input)
          print("Predicted label: ", label.numpy())

    def save_model(self,path):
        self.export_model.save(path)

    def load_model(self, path):
        self.lmodel = tf.keras.models.load_model(path)

    def predict(self, sentence):
        #test
        inputs = [
            sentence
        ]
        predicted_scores = self.lmodel.predict(inputs)
        predicted_int_labels = tf.argmax(predicted_scores, axis=1)
        predicted_labels = tf.gather(self.classes, predicted_int_labels)
     
        for input, label, num in zip(inputs, predicted_labels, predicted_int_labels):
          print("Question: ", input)
          print("Predicted label: ", label.numpy())
          print("num :" + str(num.numpy()))
          
    def predict_one(self, sentence):
        #test
        inputs = [
            sentence
        ]
        predicted_scores = self.lmodel.predict(inputs)
        predicted_int_labels = tf.argmax(predicted_scores, axis=1)
        print(self.classes[predicted_int_labels.numpy()[0]])

        return predicted_int_labels.numpy()[0]
          

    def get_string_labels(self, predicted_scores_batch):
          predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
          predicted_labels = tf.gather(self.classes, predicted_int_labels)
          return predicted_labels

    def int_vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.int_vectorize_layer(text), label

    
    def configure_dataset(self, dataset):
        return dataset.cache().prefetch(buffer_size=self.AUTOTUNE)


myvocal = MyVocal(32)
#myvocal.init_model()
#myvocal.create_model()
#myvocal.save_model("model/vocalmodel")
myvocal.load_model("model/vocalmodel")
myvocal.predict("on debute une partie")
myvocal.predict("met le filtre sobel")
myvocal.predict("je te connais")
myvocal.predict("qu-est ce qu on fait")
