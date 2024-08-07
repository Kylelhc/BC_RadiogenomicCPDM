import os
import random
import statistics
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, Flatten, Dense, 
                                     Reshape, LeakyReLU, BatchNormalization, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found for TensorFlow, using CPU.")


# Device configuration for PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# File paths
rootPath = os.getcwd()
SideImgPath = os.path.abspath('../BreastCancer/data/SidePNG128')
SideImgIDpath = find_file('SideViewBTF.csv', rootPath)
multiOmicPath = find_file('BTF_features.csv', rootPath)
geneExprPath = find_file('TCGABRCA_15gxp.csv', rootPath)

# Importing custom functions
from utils.geneExpr import prepareGeneDataset
from utils.multiOmic import prepareDataset, get_dataloader

def run(testset=[], indx=0, verbose=0):
    print(f'------------- Folder {indx} -------------')

    condition_train, condition_test, imgs_train, imgs_test = prepareGeneDataset(
        SideImgPath, SideImgIDpath, geneExprPath, device, test=testset
    )

    batchsize = 54
    train_loader, test_loader = get_dataloader(batchsize, condition_train, condition_test, imgs_train, imgs_test)

    # Create mock data
    num_samples = 58
    image_shape = (128, 128, 1)
    omic_features = 15

    mris, omics = next(iter(train_loader))
    mris = mris.permute(0, 2, 3, 1).cpu().numpy()
    omics = omics.view(omics.size(0), omic_features).cpu().numpy()

    # Standardize the omic data
    scaler = StandardScaler()
    omics = scaler.fit_transform(omics)

    # Define the VAE model
    input_shape = mris.shape[1:]
    omic_input_shape = omics.shape[1]

    # Encoder
    def build_encoder(input_shape, omic_input_shape, latent_dim):
        mri_input = Input(shape=input_shape)
        omic_input = Input(shape=(omic_input_shape,))
        
        x = Conv2D(32, 3, activation='relu', padding='same')(mri_input)
        x = Conv2D(64, 3, activation='relu', padding='same', strides=(2, 2))(x)
        x = Conv2D(128, 3, activation='relu', padding='same', strides=(2, 2))(x)
        x = Conv2D(256, 3, activation='relu', padding='same', strides=(2, 2))(x)
        x = Flatten()(x)
        
        combined = Concatenate()([x, omic_input])
        x = Dense(128, activation='relu')(combined)
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        
        return Model([mri_input, omic_input], [z_mean, z_log_var], name='encoder')

    # Sampling layer
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Decoder
    def build_decoder(latent_dim, output_shape):
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(16 * 16 * 256, activation='relu')(latent_inputs)
        x = Reshape((16, 16, 256))(x)
        x = Conv2DTranspose(256, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(128, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        
        return Model(latent_inputs, x, name='decoder')

    latent_dim = 2
    encoder = build_encoder(input_shape, omic_input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    mri_input = Input(shape=input_shape)
    omic_input = Input(shape=(omic_input_shape,))
    z_mean, z_log_var = encoder([mri_input, omic_input])
    z = Sampling()([z_mean, z_log_var])
    reconstructed = decoder(z)

    vae = Model([mri_input, omic_input], reconstructed, name='vae')

    # Loss function
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(tf.keras.backend.flatten(mri_input), tf.keras.backend.flatten(reconstructed))
    )
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    save_path = f'revision/VAE_models/vae_model_gene_Folder_{indx}.h5'
    vae.fit([mris, omics], mris, epochs=70, batch_size=27, verbose=verbose)
    vae.save(save_path)

    vae = tf.keras.models.load_model(save_path, custom_objects={'Sampling': Sampling, 'kl_loss': kl_loss})

    encoder = vae.get_layer('encoder')
    decoder = vae.get_layer('decoder')

    def generate_images(omics_data, num_images=1):
        omics_data = scaler.transform(omics_data)
        z_mean, z_log_var = encoder.predict([np.zeros((num_images, *image_shape)), omics_data], verbose=verbose)
        z = Sampling()([z_mean, z_log_var])
        generated_images = decoder.predict(z, verbose=verbose)
        return generated_images

    return vae, encoder, decoder, generate_images


