# Fashion-MNIST-Dataset-with-Keras-Deep Learning

We train a simple Convolutional Neural Network (CNN) with Keras on the Fashion MNIST dataset, enabling you to classify fashion images and categories.


The Fashion MNIST dataset is meant to be a (slightly more challenging) drop-in replacement for the (less challenging) MNIST dataset.


Similar to the MNIST digit dataset, the Fashion MNIST dataset includes:

60,000 training examples

10,000 testing examples

10 classes

28×28 grayscale/single channel images

The ten fashion class labels include:

0 - T-shirt/top
1 - Trouser/pants
2 - Pullover shirt
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle boot

Our project today is rather straightforward — we’re reviewing two Python files:

pyimagesearch/minivggnet.py : Contains a simple CNN based on VGGNet.

fashion_mnist.py : Our training script for Fashion MNIST classification with Keras and deep learning. This script will load the data (remember, it is built into Keras), and train our MiniVGGNet model. A classification report and montage will be generated upon training completion.

Today we’ll be defining a very simple Convolutional Neural Network to train on the Fashion MNIST dataset.

We’ll call this CNN “MiniVGGNet” since:

The model is inspired by its bigger brother, VGGNet
The model has VGGNet characteristics, including:
Only using 3×3 CONV filters
Stacking multiple CONV layers before applying a max-pooling operation
We’ve used the MiniVGGNet model before a handful of times on the PyImageSearch blog but we’ll briefly review it here today as a matter of completeness.




Our MiniVGGNet class and its build method are defined here. The build function accepts four parameters:

width : Image width in pixels.

height : Image height in pixels.

depth : Number of channels. Typically for color this value is 3 and for grayscale it is 1 (the Fashion MNIST dataset is grayscale).

classes : The number of types of fashion articles we can recognize. The number of classes affects the final fully-connected output layer. For the Fashion MNIST dataset there are a total of 10 classes.

Our model is initialized using the Sequential API.

From there, our inputShape is defined (Line 18). We’re going to use "channels_last" ordering since our backend is TensorFlow.

# Add Layers to CNN - 

Click here to download the source code to this post
In this tutorial, you will learn how to train a simple Convolutional Neural Network (CNN) with Keras on the Fashion MNIST dataset, enabling you to classify fashion images and categories.

The Fashion MNIST dataset is meant to be a (slightly more challenging) drop-in replacement for the (less challenging) MNIST dataset.

Similar to the MNIST digit dataset, the Fashion MNIST dataset includes:

60,000 training examples
10,000 testing examples
10 classes
28×28 grayscale/single channel images
The ten fashion class labels include:

T-shirt/top
Trouser/pants
Pullover shirt
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
Throughout this tutorial, you will learn how to train a simple Convolutional Neural Network (CNN) with Keras on the Fashion MNIST dataset, giving you not only hands-on experience working with the Keras library but also your first taste of clothing/fashion classification.

To learn how to train a Keras CNN on the Fashion MNIST dataset, just keep reading!


Looking for the source code to this post?
JUMP RIGHT TO THE DOWNLOADS SECTION 
Fashion MNIST with Keras and Deep Learning
2020-06-11 Update: This blog post is now TensorFlow 2+ compatible!

In the first part of this tutorial, we will review the Fashion MNIST dataset, including how to download it to your system.

From there we’ll define a simple CNN network using the Keras deep learning library.

Finally, we’ll train our CNN model on the Fashion MNIST dataset, evaluate it, and review the results.

Let’s go ahead and get started!

The Fashion MNIST dataset

Figure 1: The Fashion MNIST dataset was created by e-commerce company, Zalando, as a drop-in replacement for MNIST Digits. It is a great dataset to practice with when using Keras for deep learning. (image source)
The Fashion MNIST dataset was created by e-commerce company, Zalando.

As they note on their official GitHub repo for the Fashion MNIST dataset, there are a few problems with the standard MNIST digit recognition dataset:

It’s far too easy for standard machine learning algorithms to obtain 97%+ accuracy.
It’s even easier for deep learning models to achieve 99%+ accuracy.
The dataset is overused.
MNIST cannot represent modern computer vision tasks.
Zalando, therefore, created the Fashion MNIST dataset as a drop-in replacement for MNIST.

The Fashion MNIST dataset is identical to the MNIST dataset in terms of training set size, testing set size, number of class labels, and image dimensions:

60,000 training examples
10,000 testing examples
10 classes
28×28 grayscale images
If you’ve ever trained a network on the MNIST digit dataset then you can essentially change one or two lines of code and train the same network on the Fashion MNIST dataset!

How to install TensorFlow/Keras
To configure your system for this tutorial, I first recommend following either of these tutorials:

How to install TensorFlow 2.0 on Ubuntu
How to install TensorFlow 2.0 on macOS
Either tutorial will help you configure you system with all the necessary software for this blog post in a convenient Python virtual environment.

Please note that PyImageSearch does not recommend or support Windows for CV/DL projects.

Obtaining the Fashion MNIST dataset

Figure 2: The Fashion MNIST dataset is built right into Keras. Alternatively, you can download it from GitHub. (image source)
There are two ways to obtain the Fashion MNIST dataset.

If you are using the TensorFlow/Keras deep learning library, the Fashion MNIST dataset is actually built directly into the datasets module:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
from tensorflow.keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
Otherwise, if you are using another deep learning library you can download it directory from the the official Fashion MNIST GitHub repo.

A big thanks to Margaret Maynard-Reid for putting together the awesome illustration in Figure 2.

Project structure
To follow along, be sure to grab the “Downloads” for today’s blog post.

Once you’ve unzipped the files, your directory structure will look like this:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   └── minivggnet.py
├── fashion_mnist.py
└── plot.png
1 directory, 4 files
Our project today is rather straightforward — we’re reviewing two Python files:

pyimagesearch/minivggnet.py : Contains a simple CNN based on VGGNet.
fashion_mnist.py : Our training script for Fashion MNIST classification with Keras and deep learning. This script will load the data (remember, it is built into Keras), and train our MiniVGGNet model. A classification report and montage will be generated upon training completion.
Defining a simple Convolutional Neural Network (CNN)
Today we’ll be defining a very simple Convolutional Neural Network to train on the Fashion MNIST dataset.

We’ll call this CNN “MiniVGGNet” since:

The model is inspired by its bigger brother, VGGNet
The model has VGGNet characteristics, including:
Only using 3×3 CONV filters
Stacking multiple CONV layers before applying a max-pooling operation
We’ve used the MiniVGGNet model before a handful of times on the PyImageSearch blog but we’ll briefly review it here today as a matter of completeness.

Open up a new file, name it minivggnet.py, and insert the following code:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
Our Keras imports are listed on Lines 2-10. Our Convolutional Neural Network model is relatively simple, but we will be taking advantage of batch normalization and dropout which are two methods I nearly always recommend. For further reading please take a look at Deep Learning for Computer Vision with Python.

Our MiniVGGNet class and its build method are defined on Lines 12-14. The build function accepts four parameters:

width : Image width in pixels.
height : Image height in pixels.
depth : Number of channels. Typically for color this value is 3 and for grayscale it is 1 (the Fashion MNIST dataset is grayscale).
classes : The number of types of fashion articles we can recognize. The number of classes affects the final fully-connected output layer. For the Fashion MNIST dataset there are a total of 10 classes.
Our model is initialized on Line 17 using the Sequential API.

From there, our inputShape is defined (Line 18). We’re going to use "channels_last" ordering since our backend is TensorFlow, but in case you’re using a different backend, Lines 23-25 will accommodate.


# Now let’s add our layers to the CNN:


Our model has two sets of (CONV => RELU => BN) * 2 => POOL layers. These layer sets also include batch normalization and dropout.

Convolutional layers, including their parameters, are described in detail in this previous post.

Pooling layers help to progressively reduce the spatial dimensions of the input volume.

Batch normalization, as the name suggests, seeks to normalize the activations of a given input volume before passing it into the next layer. It has been shown to be effective at reducing the number of epochs required to train a CNN at the expense of an increase in per-epoch time.

Dropout is a form of regularization that aims to prevent overfitting. Random connections are dropped to ensure that no single node in the network is responsible for activating when presented with a given pattern.

What follows is a fully-connected layer and softmax classifier. The softmax classifier is used to obtain output classification probabilities.

The model is then returned.


Click here to download the source code to this post
In this tutorial, you will learn how to train a simple Convolutional Neural Network (CNN) with Keras on the Fashion MNIST dataset, enabling you to classify fashion images and categories.

The Fashion MNIST dataset is meant to be a (slightly more challenging) drop-in replacement for the (less challenging) MNIST dataset.

Similar to the MNIST digit dataset, the Fashion MNIST dataset includes:

60,000 training examples
10,000 testing examples
10 classes
28×28 grayscale/single channel images
The ten fashion class labels include:

T-shirt/top
Trouser/pants
Pullover shirt
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
Throughout this tutorial, you will learn how to train a simple Convolutional Neural Network (CNN) with Keras on the Fashion MNIST dataset, giving you not only hands-on experience working with the Keras library but also your first taste of clothing/fashion classification.

To learn how to train a Keras CNN on the Fashion MNIST dataset, just keep reading!


Looking for the source code to this post?
JUMP RIGHT TO THE DOWNLOADS SECTION 
Fashion MNIST with Keras and Deep Learning
2020-06-11 Update: This blog post is now TensorFlow 2+ compatible!

In the first part of this tutorial, we will review the Fashion MNIST dataset, including how to download it to your system.

From there we’ll define a simple CNN network using the Keras deep learning library.

Finally, we’ll train our CNN model on the Fashion MNIST dataset, evaluate it, and review the results.

Let’s go ahead and get started!

The Fashion MNIST dataset

Figure 1: The Fashion MNIST dataset was created by e-commerce company, Zalando, as a drop-in replacement for MNIST Digits. It is a great dataset to practice with when using Keras for deep learning. (image source)
The Fashion MNIST dataset was created by e-commerce company, Zalando.

As they note on their official GitHub repo for the Fashion MNIST dataset, there are a few problems with the standard MNIST digit recognition dataset:

It’s far too easy for standard machine learning algorithms to obtain 97%+ accuracy.
It’s even easier for deep learning models to achieve 99%+ accuracy.
The dataset is overused.
MNIST cannot represent modern computer vision tasks.
Zalando, therefore, created the Fashion MNIST dataset as a drop-in replacement for MNIST.

The Fashion MNIST dataset is identical to the MNIST dataset in terms of training set size, testing set size, number of class labels, and image dimensions:

60,000 training examples
10,000 testing examples
10 classes
28×28 grayscale images
If you’ve ever trained a network on the MNIST digit dataset then you can essentially change one or two lines of code and train the same network on the Fashion MNIST dataset!

How to install TensorFlow/Keras
To configure your system for this tutorial, I first recommend following either of these tutorials:

How to install TensorFlow 2.0 on Ubuntu
How to install TensorFlow 2.0 on macOS
Either tutorial will help you configure you system with all the necessary software for this blog post in a convenient Python virtual environment.

Please note that PyImageSearch does not recommend or support Windows for CV/DL projects.

Obtaining the Fashion MNIST dataset

Figure 2: The Fashion MNIST dataset is built right into Keras. Alternatively, you can download it from GitHub. (image source)
There are two ways to obtain the Fashion MNIST dataset.

If you are using the TensorFlow/Keras deep learning library, the Fashion MNIST dataset is actually built directly into the datasets module:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
from tensorflow.keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
Otherwise, if you are using another deep learning library you can download it directory from the the official Fashion MNIST GitHub repo.

A big thanks to Margaret Maynard-Reid for putting together the awesome illustration in Figure 2.

Project structure
To follow along, be sure to grab the “Downloads” for today’s blog post.

Once you’ve unzipped the files, your directory structure will look like this:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   └── minivggnet.py
├── fashion_mnist.py
└── plot.png
1 directory, 4 files
Our project today is rather straightforward — we’re reviewing two Python files:

pyimagesearch/minivggnet.py : Contains a simple CNN based on VGGNet.
fashion_mnist.py : Our training script for Fashion MNIST classification with Keras and deep learning. This script will load the data (remember, it is built into Keras), and train our MiniVGGNet model. A classification report and montage will be generated upon training completion.
Defining a simple Convolutional Neural Network (CNN)
Today we’ll be defining a very simple Convolutional Neural Network to train on the Fashion MNIST dataset.

We’ll call this CNN “MiniVGGNet” since:

The model is inspired by its bigger brother, VGGNet
The model has VGGNet characteristics, including:
Only using 3×3 CONV filters
Stacking multiple CONV layers before applying a max-pooling operation
We’ve used the MiniVGGNet model before a handful of times on the PyImageSearch blog but we’ll briefly review it here today as a matter of completeness.

Open up a new file, name it minivggnet.py, and insert the following code:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
Our Keras imports are listed on Lines 2-10. Our Convolutional Neural Network model is relatively simple, but we will be taking advantage of batch normalization and dropout which are two methods I nearly always recommend. For further reading please take a look at Deep Learning for Computer Vision with Python.

Our MiniVGGNet class and its build method are defined on Lines 12-14. The build function accepts four parameters:

width : Image width in pixels.
height : Image height in pixels.
depth : Number of channels. Typically for color this value is 3 and for grayscale it is 1 (the Fashion MNIST dataset is grayscale).
classes : The number of types of fashion articles we can recognize. The number of classes affects the final fully-connected output layer. For the Fashion MNIST dataset there are a total of 10 classes.
Our model is initialized on Line 17 using the Sequential API.

From there, our inputShape is defined (Line 18). We’re going to use "channels_last" ordering since our backend is TensorFlow, but in case you’re using a different backend, Lines 23-25 will accommodate.

Now let’s add our layers to the CNN:

→ Launch Jupyter Notebook on Google Colab
Fashion MNIST with Keras and Deep Learning
		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model
Our model has two sets of (CONV => RELU => BN) * 2 => POOL layers (Lines 28-46). These layer sets also include batch normalization and dropout.

Convolutional layers, including their parameters, are described in detail in this previous post.

Pooling layers help to progressively reduce the spatial dimensions of the input volume.

Batch normalization, as the name suggests, seeks to normalize the activations of a given input volume before passing it into the next layer. It has been shown to be effective at reducing the number of epochs required to train a CNN at the expense of an increase in per-epoch time.

Dropout is a form of regularization that aims to prevent overfitting. Random connections are dropped to ensure that no single node in the network is responsible for activating when presented with a given pattern.

What follows is a fully-connected layer and softmax classifier (Lines 49-57). The softmax classifier is used to obtain output classification probabilities.

The model is then returned.




Link of Downloading a Dataset - 

https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/
