# Brain_Tumor_Detection-
# Brain Tumor Detection Using CNN
## Project Overview
This project focuses on developing a deep learning model for automated brain tumor detection using Convolutional Neural Networks (CNNs). Brain tumors pose a serious risk to human health, and early detection can significantly improve treatment outcomes. This project aims to leverage deep learning techniques to classify brain MRI images as either tumor or non-tumor, providing a tool for healthcare professionals to assist in diagnosis.
## Objective
The primary goal of this project is to build a classification model that accurately identifies brain tumors from MRI scans. The model helps in distinguishing between images that contain tumors and those that do not.
## Dataset
The dataset used for this project consists of MRI images that are labeled as either containing a tumor or not. It includes various types of brain tumors (e.g., glioma, meningioma, and pituitary tumor) for classification purposes. The images are grayscale and have been pre-processed to ensure consistency in dimensions and quality.
### Data Source
<a href>https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri</a> 
## Methodology
1. Data Preprocessing: The MRI images are resized to a uniform dimension (e.g., 128x128 pixels) to be fed into the CNN.
2. Model Architecture: The CNN model is designed with multiple convolutional layers followed by max-pooling and activation functions (ReLU). The architecture captures spatial features in the MRI images, and the final fully connected layers perform classification.
   a. Convolutional Layers: For feature extraction.
   b. Pooling Layers: To reduce spatial dimensions and computational complexity.
   c. Activation Functions: ReLU for introducing non-linearity.
   d. Dropout: Applied to prevent overfitting.
   e. Output Layer: Softmax or sigmoid activation for binary classification.
3. Training: The model is trained using the cross-entropy loss function and optimized with the Adam optimizer. The training process involves dividing the dataset into training and validation sets, with metrics like accuracy and loss being monitored during the process.
4. Evaluation: The model is evaluated on a held-out test set, and performance is measured using metrics such as:
a. Accuracy: The percentage of correct predictions.
## Results
The CNN model achieved high accuracy on the test dataset, making it a viable tool for assisting in brain tumor detection.
## Technologies and Tools
a. Python: Core programming language.
b. TensorFlow/Keras: For building and training the CNN model.
c.  NumPy, Pandas: For data manipulation.
d.  Matplotlib/Seaborn: For visualizing the data and results.
e.  OpenCV: For image processing and augmentation.
f.  Scikit-learn: For model evaluation metrics like confusion matrix, precision, recall, etc.
## Future Improvements
a. Adding support for multi-class classification to identify different types of brain tumors.
b. Improving the model's generalization by incorporating more diverse MRI datasets.
c. Deployment of the model as a web application for real-time usage in hospitals.
## Conclusion
This project demonstrates the effectiveness of CNNs in detecting brain tumors from MRI scans. It provides a starting point for building advanced healthcare tools that assist in early diagnosis and treatment planning.
