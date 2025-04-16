# Indian-Sign-Language-to-Text-Conversion

The Indian Sign Language has been developed by Indiansignlanguage.org which offers a huge collection of Indian Sign Language (ISL) signs. Each sign has an image, running video and threaded discussions. This project aims to correctly recognize the gesture using a web application which is displayed by via a camera. The dataset used is a combination of ISL-CSLTR Frame Level Words and Images added by us taken using a  DSLR camera. 

This project aims to convert Indian Sign Language (ISL) gestures into readable text using both traditional machine learning and deep learning approaches. The application is designed to assist communication by interpreting hand gestures captured via a webcam or camera.

🧠 **Models Used**

✅ 1. Random Forest + MediaPipe

-Uses MediaPipe Hands for hand landmark extraction.

-Extracted (x, y) coordinates from 21 landmarks per hand.

-Landmark vectors are padded and used to train a Random Forest Classifier.

-Achieved high validation accuracy for static gesture classification with minimal computational resources.

-Serialized as .pkl model.

✅ 2. MobileNetV2 CNN Model

-Bounding box created using MediaPipe around detected hands.

-Hand images preprocessed and resized to 224x224.

-Feature extraction via pre-trained MobileNetV2 (frozen weights).

-Classification using a fully connected softmax layer.

-Trained on a processed dataset and serialized as .h5 model.

🛠️ **Technologies Used**

-Python, OpenCV, TensorFlow/Keras, scikit-learn

-MediaPipe for hand landmark detection

-Albumentations for data augmentation

-Matplotlib / Seaborn for visualization

-GitHub / Google Colab / Jupyter Notebooks
