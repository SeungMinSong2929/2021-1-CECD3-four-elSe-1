
import keras
from src.utils import split
import tensorflow as tf
import numpy as np

"""

 autoencoder.py  (author: Anson Wong / git: ankonzoid)

"""


class AutoEncoder():

    def __init__(self, modelName, info):
        self.modelName = modelName
        self.info = info
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    # Train
    def fit(self, X, n_epochs=50, batch_size=256):
        indices_fracs = split(fracs=[0.9, 0.1], N=len(X), seed=0)
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        self.autoencoder.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid))

    # Inference
    def predict(self, X):
        return self.encoder.predict(X)

    # Set neural network architecture
    def set_arch(self):

        shape_img = self.info["shape_img"]
        shape_img_flattened = (np.prod(list(shape_img)),)

        # Set encoder and decoder graphs
        if self.modelName == "simpleAE":
            encode_dim = 128

            input = keras.Input(shape=shape_img_flattened)
            encoded = keras.layers.Dense(encode_dim, activation='relu')(input)

            decoded = keras.layers.Dense(
                shape_img_flattened[0], activation='sigmoid')(encoded)

        elif self.modelName == "convAE":
            n_hidden_1, n_hidden_2, n_hidden_3 = 16, 8, 8
            convkernel = (3, 3)  # convolution kernel
            poolkernel = (2, 2)  # pooling kernel

            input = keras.layers.Input(shape=shape_img)  # (512, 512)
            x = keras.layers.Conv2D(
                n_hidden_1, convkernel, activation='relu', padding='same')(input)  # (512, 512)
            x = keras.layers.MaxPooling2D(poolkernel, padding='same')(
                x)                              # (256, 256)
            x = keras.layers.Conv2D(
                n_hidden_2, convkernel, activation='relu', padding='same')(x)     # (256, 256)
            x = keras.layers.MaxPooling2D(poolkernel, padding='same')(
                x)                              # (128, 128)
            x = keras.layers.Conv2D(
                n_hidden_3, convkernel, activation='relu', padding='same')(x)     # (128, 128)
            encoded = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(
                x)                        # (64, 64)

            x = keras.layers.Conv2D(
                n_hidden_3, convkernel, activation='relu', padding='same')(encoded)  # (64, 64)
            x = keras.layers.UpSampling2D(poolkernel)(
                x)                                                # (128, 128)
            x = keras.layers.Conv2D(
                n_hidden_2, convkernel, activation='relu', padding='same')(x)       # (128, 128)
            x = keras.layers.UpSampling2D(poolkernel)(
                x)                                                # (256,256)
            x = keras.layers.Conv2D(
                n_hidden_1, convkernel, activation='relu', padding="same")(x)        # (254,254)
            x = keras.layers.UpSampling2D(poolkernel)(x)
            decoded = keras.layers.Conv2D(
                shape_img[2], convkernel, activation='sigmoid', padding='same')(x)

        elif self.modelName == "stackedAE":
            input = keras.layers.Input(shape=shape_img)

            # encoder
            x = keras.layers.Conv2D(64, kernel_size=(
                3, 3), activation="relu", padding="same")(input)
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same")(x)
            x = keras.layers.Conv2D(128, kernel_size=(
                3, 3), strides=1, activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same")(x)
            x = keras.layers.Conv2D(256, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same")(x)
            x = keras.layers.Conv2D(512, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same")(x)
            x = keras.layers.Conv2D(512, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            encoded = keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2, padding="same")(x)

            # decoder
            x = keras.layers.Conv2D(512, kernel_size=(
                3, 3), activation="relu", padding="same")(encoded)
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = keras.layers.Conv2D(512, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = keras.layers.Conv2D(256, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            x = keras.layers.UpSampling1D((2, 2))(x)
            x = keras.layers.Conv2D(128, kernel_size=(
                3, 3), activation="relu", padding="same")(x)
            x = keras.layers.UpSampling1D((2, 2))(x)
            x = keras.layers.Conv2D(64, kernel_size=(
                3, 3),  activation="relu", padding="same")(x)
            x = keras.layers.UpSampling1D((2, 2))(x)
            decoded = keras.layers.Conv2D(3, kernel_size=(
                3, 3), padding="same", activation="sigmoid")(x)
        else:
            raise Exception("Invalid model name given!")

        # Create autoencoder model
        autoencoder = keras.Model(input, decoded)
        input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
        output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]

        # Create encoder model
        encoder = keras.Model(input, encoded)  # set encoder
        input_encoder_shape = encoder.layers[0].input_shape[1:]
        output_encoder_shape = encoder.layers[-1].output_shape[1:]

        # Create decoder model
        decoded_input = keras.Input(shape=output_encoder_shape)
        if self.modelName == 'simpleAE':
            # single layer
            decoded_output = autoencoder.layers[-1](decoded_input)
        elif self.modelName == 'convAE':
            decoded_output = autoencoder.layers[-7](decoded_input)  # Conv2D
            # UpSampling2D
            decoded_output = autoencoder.layers[-6](decoded_output)
            decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
            # UpSampling2D
            decoded_output = autoencoder.layers[-4](decoded_output)
            decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D
            # UpSampling2D
            decoded_output = autoencoder.layers[-2](decoded_output)
            decoded_output = autoencoder.layers[-1](decoded_output)  # Conv2D
        elif self.modelName == "stackedAE":
            decoded_output = autoencoder.layers[-11](decoded_input)
            decoded_output = autoencoder.layers[-9](decoded_input)
            decoded_output = autoencoder.layers[-7](decoded_input)
            decoded_output = autoencoder.layers[-6](decoded_input)
            decoded_output = autoencoder.layers[-7](decoded_input)
            decoded_output = autoencoder.layers[-4](decoded_input)
            decoded_output = autoencoder.layers[-3](decoded_input)
            decoded_output = autoencoder.layers[-2](decoded_input)
            decoded_output = autoencoder.layers[-1](decoded_input)

        else:
            raise Exception("Invalid model name given!")
        decoder = keras.Model(decoded_input, decoded_output)
        decoder_input_shape = decoder.layers[0].input_shape[1:]
        decoder_output_shape = decoder.layers[-1].output_shape[1:]

        # Generate summaries
        print("\nautoencoder.summary():")
        print(autoencoder.summary())
        print("\nencoder.summary():")
        print(encoder.summary())
        print("\ndecoder.summary():")
        print(decoder.summary())

        # Assign models
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

    # Compile
    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    # Load model architecture and weights
    def load_models(self, loss="binary_crossentropy", optimizer="adam"):
        print("Loading models...")
        self.autoencoder = keras.models.load_model(
            self.info["autoencoderFile"])
        self.encoder = keras.models.load_model(self.info["encoderFile"])
        self.decoder = keras.models.load_model(self.info["decoderFile"])
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)
        self.decoder.compile(optimizer=optimizer, loss=loss)

    # Save model architecture and weights to file
    def save_models(self):
        print("Saving models...")
        self.autoencoder.save(self.info["autoencoderFile"])
        self.encoder.save(self.info["encoderFile"])
