# Image_Content_Classifier_Integrated_with_Streamlit
** This Repository  is invloved with NSFW datasets which is used for research and educational purpose only**
- source file for adult content from [github](https://github.com/EBazarov/nsfw_data_source_urls/tree/master) 
- source file for violence and safe img  from [kaggle](https://www.kaggle.com/datasets/khushhalreddy/violence-detection-dataset)

# Image Content Classification Model

## Objective

The goal of this project is to develop a machine learning model to classify images into three categories: 'Violent', 'Adult Content', and 'Safe'. The model is then integrated into a Streamlit app to demonstrate its functionality in a live web interface.

## Tools and Libraries

- **Python**: Main programming language.
- **TensorFlow/Keras**: For building and training the convolutional neural network (CNN).
- **NumPy**: For data manipulation and preprocessing.
- **Matplotlib**: For creating visualizations.
- **Streamlit**: To create an interactive web app showcasing the model.

## Steps

### 1. Environment Setup

Set up a Python environment and install the necessary libraries using pip:
```sh

   pip install tensorflow numpy matplotlib streamlit pillow opencv-python

###  2. Data Collection & Preprocessing

import os
import requests

#Function to download images from URLs
def download_images_from_urls(url_file, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(url_file, 'r') as file:
        urls = file.readlines()
    
    for idx, url in enumerate(tqdm(urls, desc="Downloading adult content images")):
        try:
            response = requests.get(url.strip(), timeout=10)
            response.raise_for_status()
            with open(os.path.join(save_dir, f"adult_{idx}.jpg"), 'wb') as img_file:
                img_file.write(response.content)
        except Exception as e:
            print(f"Failed to download {url.strip()}: {e}")

# Example usage
url_file = 'urls.txt'
save_dir = 'Assignment/violence/dataset/train/adult_content'
download_images_from_urls(url_file, save_dir)

### 3 . Model Development & Training
*Data Augmentation and Preprocessing*
To increase the diversity of the training data, we applied data augmentation techniques such as rotation, zooming, and flipping. This helps in making the model more robust.

### 4 Model Evaluation
The model was evaluated using standard metrics such as accuracy, precision, recall, and F1-score to determine its performance. These metrics helped in understanding how well the model was able to classify the images into the defined categories.

###  5. Streamlit Integration
The trained model was integrated into a Streamlit web app. The app allows users to upload images, which are then processed and classified by the model in real-time. The app also displays the classification results and provides interactive elements to show model accuracy and other statistics
