import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models



class MyAudio:

    def __init__(self, datatsel_url):

        self.data_dir = datatsel_url
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.AUTOTUNE = tf.data.AUTOTUNE

        self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        self.commands = self.commands[self.commands != 'README.md']
        print('Commands:', self.commands)



    def init_model(self, bsz):
                
        self.filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
        self.filenames = tf.random.shuffle(self.filenames)
        self.num_samples = len(self.filenames)
       
        #/!\-------------------------------------------/!\
        #[(+)]don't forget to adapt the number of file [(+)]
        #/!\-------------------------------------------/!\
        self.train_files = self.filenames[:55]
        self.val_files = self.filenames[55: 10]
        self.test_files = self.filenames[-10:]
        
        self.files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)

        self.waveform_ds = self.files_ds.map(
            map_func=self.get_waveform_and_label,
            num_parallel_calls=self.AUTOTUNE)

        self.spectrogram_ds = self.waveform_ds.map(
          map_func=self.get_spectrogram_and_label_id,
          num_parallel_calls=self.AUTOTUNE)

        self.train_ds = self.spectrogram_ds
        self.val_ds = self.preprocess_dataset(self.val_files)
        self.test_ds = self.preprocess_dataset(self.test_files)

        self.batch_size = bsz
        self.train_ds = self.train_ds.batch(self.batch_size)
        self.val_ds = self.val_ds.batch(self.batch_size)

        self.train_ds = self.train_ds.cache().prefetch(self.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(self.AUTOTUNE)


    def create_model(self):
        for spectrogram, _ in self.spectrogram_ds.take(1):
            input_shape = spectrogram.shape

        num_labels = len(self.commands)
        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=self.spectrogram_ds.map(map_func=lambda spec, label: spec))

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        EPOCHS = 10
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
        plt.legend(['loss', 'accuracy'])
        plt.show()

    def test_model(self):
        test_audio = []
        test_labels = []

        for audio, label in self.test_ds:
          test_audio.append(audio.numpy())
          test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)


        y_pred = np.argmax(self.model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')

    def predict(self, path):

        sample_file = tf.io.gfile.glob(path)
        ans =['non', 'oui', 'qui']
        sample_ds = self.preprocess_dataset(sample_file)
        for spectrogram, label in sample_ds.batch(1):
            prediction = self.model(spectrogram)
            score = tf.nn.softmax(prediction[0])
            print(str(label[0]) + " " + str(tf.nn.softmax(prediction[0])))
            #plt.bar(self.commands, tf.nn.softmax(prediction[0]))
            #plt.title(f'Predictions for "{self.commands[label[0]]}"')
            #plt.show()
        
        print(ans[int(np.argmax(score))] + " " + str(100 * np.max(score)) + "%")
        return int(np.argmax(score))

    def save_model(self,path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def decode_audio(self, audio_binary):
        # Decode WAV-encoded audio files to `float32` tensors, normalized
        # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        # Since all the data is single channel (mono), drop the `channels`
        # axis from the array.
        return tf.squeeze(audio, axis=-1)

    def get_label(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        return parts[-2]

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=self.get_waveform_and_label,
            num_parallel_calls=self.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=self.get_spectrogram_and_label_id,
            num_parallel_calls=self.AUTOTUNE)
        return output_ds

    def get_spectrogram_and_label_id(self, audio, label):
        spectrogram = self.get_spectrogram(audio)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id

    def get_spectrogram(self, waveform):
        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = 16000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [16000] - tf.shape(waveform),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram


myaudio = MyAudio("sound")
"""myaudio.init_model(64)
myaudio.create_model()
myaudio.test_model()
myaudio.save_model("model/audiomodel")"""

myaudio.load_model("model/audiomodel")
#r = myaudio.predict("./qui.wav")
#print("ans: " + str(r))