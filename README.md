# Introduction to deep learning
[MIT's introductory program on deep learning methods](http://introtodeeplearning.com/)
# Intro to TensorFlow and Music Generation with RNNs (Lab1)
Used libraries: numpy, matplotlib.pyplot, tensorflow, mitdeeplearning, os, time, functools, tqdm
## Intro to TensorFlow
TensorFlow is a software library extensively used in machine learning. Here we'll learn how computations are represented and how to define simple neural networks in TensorFlow. TensorFlow uses a high-level API called [Keras](https://www.tensorflow.org/guide/keras?hl=ru) that provides a powerful, intuitive framework for building and training deep learning models. In this section you will learn the basics of computations in TensorFlow, the Keras API, and TensorFlow's new imperative execution style enabled by [Eager](https://blog.research.google/2017/10/eager-execution-imperative-define-by.html).
### The task:
1. Creating 0-d, 1-d, 2-d, 4-d Tensors
2. Performing simple computations and constructing computation functions using TensorFlow
3. Defining neural networks in TensorFlow
   – Defining a network Layer
   – Defining a neural network using the Sequential API
   – Defining a model using subclassing
   – Defining a model using subclassing and specifying custom behavior
4. Performing automatic differentiation in TensorFlow
   – Gradient computation with GradientTape
   – Function minimization with automatic differentiation and SGD
## Music Generation with RNNs
In the second portion of the lab, we will play around with building a Recurrent Neural Network (RNN) for music generation. We will be using a "character RNN" to predict the next character of sheet music in ABC notation. Finally, we will sample from this model to generate a brand new music file that has never been heard before!
### The task:
1. Downloading the dataset of thousands of Irish folk songs, represented in the ABC notation
2. Defining numerical representation of text-based dataset
3. Vectorizing the songs string
4. Defining the RNN model
5. Training the model: loss and training operations
6. Generating music using the RNN model
# Computer Vision (Lab 2, 3)
Used libraries: numpy, matplotlib.pyplot, tensorflow, mitdeeplearning, tqdm, random, IPython, functools
## MNIST Digit Classification
We will build and train a convolutional neural network (CNN) for classification of handwritten digits from the famous [MNIST](https://yann.lecun.com/exdb/mnist/) dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images. Our classes are the digits 0-9.
### The task:
1. Downloading the MNIST dataset and loading the dataset and display a few random samples from it
2. Building a Neural Network for handwritten digit classification
3. Building a Convolutional Neural Network (CNN) for handwritten digit classification
4. Training the model using stochastic gradient descent
## Diagnosing Bias in Facial Detection Systems
In this lab, we'll explore a prominent aspect of applied deep learning for computer vision: facial detection. Consider the task of facial detection: given an image, is it an image of a face? We will build a semi-supervised variational autoencoder (SS-VAE) that learns the latent distribution of features underlying face image datasets in order to uncover hidden biases.
### The task:
1. Downloading dataset of positive examples and a dataset of negative examples
2. Defining and training the CNN model
3. Formalizing two key aspects of the VAE model and defining relevant functions for each:
   – Defining the VAE loss function
   – Defining a function to implement the VAE sampling operation
4. Building semi-supervised VAE (SS-VAE):
   – Defining the SS-VAE loss function
   – Defining the decoder portion of the SS-VAE
   – Defining and creating the SS-VAE
   – Training the SS-VAE
5. Uncovering hidden biases through learned latent features
