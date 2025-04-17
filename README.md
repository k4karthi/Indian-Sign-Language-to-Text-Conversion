# Indian-Sign-Language-to-Text-Conversion

The Indian Sign Language has been developed by Indiansignlanguage.org which offers a huge collection of Indian Sign Language (ISL) signs. Each sign has an image, running video and threaded discussions. This project aims to correctly recognize the gesture using a web application which is displayed by via a camera. The dataset used is a combination of ISL-CSLTR Frame Level Words and Images added by us taken using a  DSLR camera. 

This project aims to convert Indian Sign Language (ISL) gestures into readable text using both traditional machine learning and deep learning approaches. The application is designed to assist communication by interpreting hand gestures captured via a webcam or camera.

![WhatsApp Image 2025-04-15 at 18 09 01_4f39023d](https://github.com/user-attachments/assets/bce33426-484e-493d-bca1-abf32634b435)

Attached is a sample of dataset created by us which is merged with some images from ISL-CSLTR images

-dataset_augmentation.ipynb is python file to increase the number of images per class and include data augmentation

-random_forest_mediapipe.ipynb , mobilenetV2.ipynb are scripts for 2 models trained on this dataset

-app_pkl_model.py , app_h5_model.py are script for running web application using streamlit. Make sure you install all the dependencies before running the script using pip command


**Models Used**

1. Random Forest + MediaPipe

-Uses MediaPipe Hands for hand landmark extraction.

-Extracted (x, y) coordinates from 21 landmarks per hand.

-Landmark vectors are padded and used to train a Random Forest Classifier.

-Achieved high validation accuracy for static gesture classification with minimal computational resources.

-Serialized as .pkl model.

2. MobileNetV2 CNN Model

-Bounding box created using MediaPipe around detected hands.

-Hand images preprocessed and resized to 224x224.

-Feature extraction via pre-trained MobileNetV2 (frozen weights).

-Classification using a fully connected softmax layer.

-Trained on a processed dataset and serialized as .h5 model.

🛠**Technologies Used**

-Python, OpenCV, TensorFlow/Keras, scikit-learn

-MediaPipe for hand landmark detection

-Albumentations for data augmentation

-Matplotlib / Seaborn for visualization

-GitHub / Google Colab / Jupyter Notebooks

 **Results**

The first image attached shows result using MobileNet model. Bounding Box was drawn around hands and then gesture label is recognized.
![Screenshot 2025-03-25 022108](https://github.com/user-attachments/assets/a3942b79-d515-4d92-b2ac-1598895a1e75)
The second image shows approach using Random Forest where coordinates extracted using mediapipe was used to recognize the gesture.
![Screenshot 2025-04-08 085956](https://github.com/user-attachments/assets/a725d081-3eb4-4acf-b100-099356881d46)

