""" Setup Environment """
import tensorflow as tf
from tensorflow.keras import metrics

import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

from transformers import TFViTModel
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50V2
from tensorflow.keras.applications.convnext import ConvNeXtBase, ConvNeXtSmall, ConvNeXtTiny

import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel


""" Constants: """
# Image size
#target_size = (224, 224, 12)
#latent_dim = 1024 # Number of latent dim parameters

""" Variational Autoencoder: """

""" VIT """
class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, backbone, **kwargs):
        super(ViTLayer, self).__init__(**kwargs)
        self.backbone = backbone
        
    def build(self, input_shape):
        self.vit = TFViTModel.from_pretrained(self.backbone)
        
    def call(self, inputs):
        out = self.vit(inputs)['pooler_output']
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vit.config.hidden_size)

    
def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = K.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random


def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch #*target_size[0]*target_size[1]
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        #return kl_loss_batch*(-.5)
        return kl_loss_batch*(-.5)#(-5e-4)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch
    
    return total_loss


def get_Variational_Autoencoder(model_path=None, backbone=None, target_size = (224, 224, 12), latent_dim = 1024, optimizer='adam', lr=0.001, model_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()], encoder_backbone=None):
    
    """ Encoder Part: """
    input_data = Input(shape=target_size, name='encoder_input')

    if not encoder_backbone:
        # Conv block 1
        encoder = Conv2D(32, 3, activation="relu", strides=2, padding="same", name="conv_1")(input_data)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)

        # Conv block 2
        encoder = Conv2D(64, 3, activation="relu", strides=2, padding="same", name="conv_2")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)

        # Conv block 3
        encoder = Conv2D(128, 3, activation="relu", strides=2, padding="same", name="conv_3")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)
        conv_shape = K.int_shape(encoder) #Shape of conv to be provided to decoder

    elif encoder_backbone == 'vit':
        # ViT uses chanel first, but our images has chanel last
        chanel_first_inputs = tf.keras.layers.Permute((3,1,2))(input_data)
        vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")(chanel_first_inputs)
        encoder = vit['pooler_output']#['last_hidden_state']
        conv_shape = (None, 28, 28, 128)
        
    elif encoder_backbone == 'ResNet50V2':
        if target_size[2] != 3:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights='imagenet')(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
                
    elif encoder_backbone == 'ConvNeXtBase':
        if target_size[2] != 3:
            cnn = ConvNeXtBase(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ConvNeXtBase(input_shape=target_size, include_top=False, weights=None)(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
        
    elif encoder_backbone == 'ConvNeXtTiny':
        if target_size[2] != 3:
            cnn = ConvNeXtTiny(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ConvNeXtTiny(input_shape=target_size, include_top=False, weights=None)(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
    
    encoder = Flatten()(encoder)
        
    """ Latent Distribution and Sampling """
    initializer = tf.keras.initializers.Zeros()

    distribution_mean = Dense(latent_dim, name='mean', kernel_initializer=initializer)(encoder)
    distribution_variance = Dense(latent_dim, name='log_variance', kernel_initializer=initializer)(encoder)

    latent_encoding = Lambda(sample_latent_features)([distribution_mean, distribution_variance])
    
    encoder_model = Model(input_data, latent_encoding)

    """ Decoder Part """
    decoder_input = Input(shape=(latent_dim), name='decoder_input')
    decoder = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(decoder_input)
    decoder = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
    
    # Transpose Conv block 1
    decoder = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same", name="deconv_1")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    # Transpose Conv block 2
    decoder = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="deconv_2")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    # Transpose Conv block 1
    decoder = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="deconv_3")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder_output = Conv2DTranspose(target_size[2], 3, activation='relu', padding="same")(decoder)
        
    decoder_model = Model(decoder_input, decoder_output)
    
    """ Model: """
    encoded = encoder_model(input_data)
    decoded = decoder_model(encoded)
    
    autoencoder = Model(input_data, decoded)
    
    if optimizer == 'adam':
        if int(tf.__version__.split('.')[1]) >= 11:
            optimizer = tf.keras.optimizers.legacy.Adam(lr=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
            
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
    
    # Compile VAE
    if metrics:
        autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer=optimizer)
        
    else:
        autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), 
                            optimizer=optimizer, 
                            metrics=model_metrics)

    if model_path:
        # Load weights
        autoencoder.load_weights(model_path, by_name=True)
    
    model = keras.Sequential()
    if backbone:
        for layer in autoencoder.layers[:-1]: # just exclude last layer from copying
            model.add(layer)
        
        model.trainable = False
        
        return model
    
    return autoencoder



class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=1024, backbone='resnet', input_shape=(3, 224, 224)):
        super(VAEEncoder, self).__init__()

        self.latent_dim = latent_dim

        if backbone == 'resnet':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            feature_dim = self.model.fc.in_features
        elif backbone == 'vit':
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            feature_dim = self.model.config.hidden_size

        # Flatten the feature dimensions for the latent vector
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(feature_dim, latent_dim * 2)

    def forward(self, x):
        if hasattr(self.model, 'fc'):  # For CNN-based models like ResNet
            x = self.model(x)
        else:  # For models like ViT
            x = self.model(pixel_values=x).last_hidden_state
            x = torch.mean(x, dim=1)  # Global average pooling equivalent for ViT

        x = self.flatten(x)
        x = self.fc(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)
        return mean, log_var



class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=1024, output_shape=(3, 224, 224), dropout_rate=0.2):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder_input = nn.Linear(latent_dim, 4096)

        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.ConvTranspose2d(32, output_shape[0], kernel_size=6, stride=2, padding=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 256, 4, 4)
        return self.deconv_blocks(z)


class VAE(nn.Module):
    def __init__(self, latent_dim=1024, output_shape=(3, 224, 224), backbone='resnet', dropout_rate=0.2):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dim, backbone)
        self.decoder = VAEDecoder(latent_dim, output_shape, dropout_rate)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var

# vae = VAE(latent_dim=1024, output_shape=(3, 224, 224), backbone='vit', dropout_rate=0.1)


def train_vae(model, dataloader, optimizer, epochs, device='cpu'):
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)  # Send data to device (CPU or GPU)
            optimizer.zero_grad()  # Zero out gradients

            # Forward pass
            recon_batch, mean, log_var = model(data)
            loss = vae_loss(recon_batch, data, mean, log_var)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')