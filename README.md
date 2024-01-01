# DF-DM: A foundational process model for multimodal data fusion in the artificial intelligence era.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains two use cases for the DF-DM model. A gender violence case for Public Health, and the case of diabetic rethinopathy for retina images.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

There are a lot of fields that can benefit from multimodal data fusion, in health case 2 common cases are the Public Health where data could be obtained from different sources, and clinical data also obtained from multiple sources. This framework leverages state-of-the-art foundational models to combine satellite imagery and social media data to understand the gender violence, and also the use of medical images and metadata for diabetic rethinopathy. The framework is flexible and can be adapted to other multimodal data fusion tasks.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8.15
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dsrestrepo/DF-DM.git
cd DF-DM
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API (Optional) key if you'll use GPT as foundational model:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```
Make sure you have a valid OpenAI API key to access the language model.

5. Set up your Sentinel Hub APIs to get the Satellite Images:

You'll get those in your profile in your sentinell hub account.

```makefile
CLIENT_ID = your_client_id
CLIENT_SECRET = your_client_secret
```


## Data

This project uses 5 datasets. You'll find instructions and code about to extract each dataset in `get_datasets.ipynb`:

1. Gender Violence Dataset: A dataset of internet data such as social media or google searches, and satellite images to predict gender violence. The codes can be used to extract a dataset for other tasks. The codes to extrac the dataset are avaibale in: `datasets/violence_prediction`.

* Satellite: To download the satellite images go to `datasets/violence_prediction/Satellite`. There you'll find the satellite extractor, this code uses the [Sentinel Hub API](https://www.sentinel-hub.com/develop/api/). Take into account that the satellite extractor requires the coordinates of the Region of Interes (ROI). You can use the file `Coordinates/get_coordinates.ipynb` to generate the ROI of your specific location. There is also a `DataAnalysis.ipynb` to assess the quality of the images.
* Metadata: The labels are located in the directory `datasets/violence_prediction/Metadata`. The labels were downloaded from open public data sources through the number of police reports of domestic violence reported in Colombia  from January 1, 2010 to August 28, 2023. You can find information about the data sources in the `data_sources.txt`. Use the `get_dataset.ipynb` to preprocess and merge the data sources, and the `Data_Analysis.ipynb` to run a data analysis.

2. [BRSET Dataset](https://physionet.org/content/brazilian-ophthalmological/1.0.0/): The Brazilian Multilabel Ophthalmological Dataset (BRSET) is a valuable resource designed to enhance scientific development and validate machine learning models in ophthalmology. With 16,266 retinal fundus images from 8,524 Brazilian patients, it includes demographic information, anatomical parameters, quality control measures, and multi-label annotations. This dataset empowers computer vision models to predict demographic characteristics and classify diseases, making it a pivotal tool for advancing research and innovation in ophthalmological machine learning.

## Usage

1. Get the dataset: Use the notebook `get_datasets.ipynb`. Functions and code to extract and preprocess each dataset were created.

2. Extract the embeddings: To extract the embeddings you can use Models with support to Open AI API such as GPT 3.5, GPT-4, or comeds with support to the llama cpp package such as LLAMA 2 7B, LLAMA 2 13B, LLAMA 2 70B, or Mistral 7B 


## Contributing
Contributions to this research project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or research.
3. Make your changes.
4. Create tests.
5. Submit a pull request.


## License
This project is licensed under the MIT License.


## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu
