from .Preprocessing.selfsupervised_data_praparation import plot_samples, get_dataset_list
from .Preprocessing.generate_embeddings import generate_embeddings_df, save_embeddings_as_csv, get_image_name, split_columns
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50V2
from tensorflow.keras.applications.convnext import ConvNeXtBase, ConvNeXtSmall, ConvNeXtTiny

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 


from .cv_data_loader import SatelliteImageFolderDataset
from .cv_models import FoundationalCVModel

from torch.utils.data import DataLoader
import torch

import os
import pandas as pd
#import joblib

import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


def get_model(model_name, model_input, latent_dim, encoder_backbone, model_path=None):
    
    # Get model:
    tf.random.set_seed(0)

    if model_name == 'variational_autoencoder':
        from .Models import Variational_Autoencoder
        model = Variational_Autoencoder.get_Variational_Autoencoder(model_path=model_path, backbone=True, target_size=model_input, latent_dim=latent_dim, lr=0.0001, encoder_backbone=encoder_backbone)
    elif model_name == 'autoencoder':
        from .Models import Autoencoder
        model = Autoencoder.get_Autoencoder(model_path=model_path, backbone=True, target_size=model_input, latent_dim=latent_dim, encoder_backbone=encoder_backbone)
    elif model_name == 'vit':
        from .Models import ViT
        model = ViT.get_vit_backbone(model_input)
    elif model_name == 'MobileNetV2':
        cnn = MobileNetV2(input_shape=model_input, include_top=False, weights='imagenet')
    elif model_name == 'VGG16': # min depth
        cnn = VGG16(input_shape=model_input, include_top=False, weights='imagenet')
    elif model_name == 'ResNet50V2':
        cnn = ResNet50V2(input_shape=model_input, include_top=False, weights='imagenet') 
    elif model_name == 'ConvNeXtTiny':
        cnn = ConvNeXtTiny(input_shape=model_input, include_top=False, weights='imagenet')  

    if model_name in ['MobileNetV2', 'VGG16', 'ResNet50V2', 'ConvNeXtTiny']:
        model = Sequential()
        model.add(cnn)
        model.add(tf.keras.layers.GlobalAveragePooling2D())

        # freeze:
        for layer in model.layers:
            layer.trainable = False

    model.summary()
    
    return model

def generate_satellite_embeddings_df(path, model_name, model_input, latent_dim, encoder_backbone, embeddings_path, model_path=None, ignore_black=False):
    
    target_size = model_input
    
    # Get list of images
    image_list = get_dataset_list(path, ignore_black=ignore_black, show_dirs=True, head=0)
    
    # Get model
    model = get_model(model_name, model_input, latent_dim, encoder_backbone, model_path)
    
    # Generate embeddings
    embeddings = generate_embeddings_df(image_list=image_list, model=model, target_size=target_size)
    
    # save embeddings to csv
    save_embeddings_as_csv(df=embeddings, path=embeddings_path)
    
    return embeddings



# Define a function to generate embeddings in parallel
def generate_embeddings(batch, batch_number, model):
    """
    Generate image embeddings for a batch of images using the specified model.

    Parameters:
    - batch (tuple): A batch of images where the first element is a list of image names, and the second element is a tensor of images.
    - batch_number (int): The batch number for tracking progress.
    - model (torch.nn.Module): The model used to generate image embeddings.

    Returns:
    tuple: A tuple containing a list of image names and their corresponding embeddings.

    Example Usage:
    ```python
    img_names, embeddings = generate_embeddings(batch, batch_number, model)
    ```

    Note:
    - This function processes a batch of images and generates embeddings for each image.
    - It is typically used in a data loading pipeline to generate embeddings for a dataset.
    """
    img_names, images = batch[0], batch[1]

    with torch.no_grad():
        features = model(images)

    if batch_number % 10 == 0:
        print(f"Processed batch number: {batch_number}")

    return img_names, features


def foundational_satellite_embeddings_df(batch_size=32, path="../BRSET/images/", dataset_name='BRSET', backbone="dinov2", directory='Embeddings', transform=None, image_files=None, save=True, embed_list=False):
    """
    Generate image embeddings and save them in a DataFrame.

    Parameters:
    - batch_size (int, optional): The batch size for processing images. Default is 32.
    - path (str, optional): The path to the folder containing the images. Default is "../BRSET/images/".
    - backbone (str, optional): The name of the foundational CV model to use. Default is "dinov2".
    - directory (str, optional): The directory to save the generated embeddings DataFrame. Default is 'Embeddings'.
    - dataset_name (str, optional): The dataset used to generate the embeddings. Default is 'BRSET'.

    Example Usage:
    ```python
    get_embeddings_df(batch_size=64, path="data/images/", dataset='BRSET', backbone="vit_base", directory='output_embeddings')
    ```

    Note:
    - This function generates image embeddings for a dataset and saves them in a DataFrame.
    - The `backbone` parameter specifies the underlying model used for feature extraction.
    - The resulting DataFrame contains image names and their corresponding embeddings.

    """
    print('#'*50, f' {backbone} ', '#'*50)
    
    # Create the custom dataset
    shape = (224, 224)
    dataset = SatelliteImageFolderDataset(folder_path=path, shape=shape, transform=transform, image_files=image_files)
        
    # Create a DataLoader to generate embeddings
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = FoundationalCVModel(backbone)

    img_names = []
    features = []
    for batch_number, batch in enumerate(dataloader, start=1):
        img_names_aux, features_aux = generate_embeddings(batch, batch_number, model)
        img_names.append(img_names_aux)
        features.append(features_aux)

    """
    # Parallelize the embedding generation process using joblib
    results = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(generate_embeddings)(batch, batch_number)
        for batch_number, batch in enumerate(dataloader, start=1)
    )
    """

    # Flatten the results to get a list of image names and their corresponding embeddings
    all_img_names = [item for sublist in img_names for item in sublist]
    all_embeddings = [item.tolist() for sublist in features for item in sublist]

    # Create a DataFrame with image names and embeddings
    df = pd.DataFrame({
        'ImageName': all_img_names,
        'Embeddings': all_embeddings
    })
    
    #print(df)
    if embed_list:
        df_aux = pd.DataFrame(df['Embeddings'].tolist())
        df = pd.concat([df['ImageName'], df_aux], axis=1)
    
    if save:
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(f'{directory}/{dataset_name}'):
            os.makedirs(f'{directory}/{dataset_name}')

        df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)
    else:
        return df

def get_foundational_satellite_embeddings_df(batch_size, path, dataset_name, backbone, directory, save=False):
    
    df = pd.DataFrame(columns=['Municipality Code', 'Date', 'Embeddings'])
    for city in os.listdir(path):
        city_path = os.path.join(path, city)
        df_city = foundational_satellite_embeddings_df(batch_size=batch_size, path=city_path, dataset_name=dataset_name, backbone=backbone, directory=directory, save=False)
        # Add city
        df_city['Municipality Code'] = city
        df_city['Date'] = df_city['ImageName'].apply(lambda x : get_image_name(os.path.join(city_path, x)))
        
        df_city = df_city[['Municipality Code', 'Date', 'Embeddings']]                                             
        
        df = pd.concat([df, df_city], ignore_index=True)
        

    df = split_columns(df, "Embeddings")
        
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(f'{directory}/{dataset_name}'):
        os.makedirs(f'{directory}/{dataset_name}')

    df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)