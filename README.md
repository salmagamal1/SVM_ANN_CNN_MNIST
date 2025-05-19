# MNIST Digit Classification using SVM, ANN, and CNN

This project presents a comparative study of three supervised learning algorithms—Support Vector Machine (SVM), Artificial Neural Network (ANN), and Convolutional Neural Network (CNN)—applied to the MNIST dataset for handwritten digit recognition.  
The goal is to evaluate and contrast the performance of these models in terms of accuracy, training time, and computational efficiency.

## Project Structure

- **Supervised_Project.ipynb**  
  A comprehensive Jupyter Notebook containing data preprocessing steps, model implementations, training processes, evaluation metrics, and visualizations for SVM, ANN, and CNN.

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a benchmark dataset in the field of machine learning and computer vision.  
It comprises 70,000 grayscale images of handwritten digits (0 through 9), each sized 28x28 pixels.  
The dataset is divided into 60,000 training images and 10,000 testing images.

## Models Implemented

### 1. Support Vector Machine (SVM)
- Utilizes a linear kernel for classification.
- Serves as a baseline model for comparison.
- Implemented using scikit-learn's `SVC` class.

### 2. Artificial Neural Network (ANN)
- A feedforward neural network with one or more hidden layers.
- Uses ReLU activation functions and dropout regularization.
- Built with TensorFlow's Keras API.

### 3. Convolutional Neural Network (CNN)
- Designed to capture spatial hierarchies in image data.
- Includes convolutional layers, pooling layers, and fully connected layers.
- Built using TensorFlow's Keras API.

#### CNN Architecture Exploration  
We implemented and evaluated **multiple CNN models** with different architectures, kernel sizes, number of filters, dropout rates, and optimizers.  
Our goal was to identify the **best-performing architecture and hyperparameter combination** for the MNIST classification task.  
This iterative experimentation process was crucial in boosting accuracy and improving generalization.

## Evaluation Metrics

- **Accuracy**: Proportion of correctly classified images.
- **Confusion Matrix**: Insight into misclassifications.
- **Training Time**: Measures computational efficiency.
- **Loss & Accuracy Curves**: Visual tools for assessing training performance.
