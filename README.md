# CNN Image Predictions: Cat or Dog? 

## Introduction 

The purpose of this repo is to utilize convolutional neural networks (CNN) on a less common and more complex set of images. This time around, I will use coloured images of cats and dogs to train and test the model. In addition, I will also be using a pre-trained model, more specifically, the MobileNet V2 model, to solve some of the problems that my model was faced with. 

## Packages and Resources Used 
Main Packages:
- Os
- Random
- Tensorflow
- Numpy
- Matplotlib
- PIL

Images downloaded from: https://www.microsoft.com/en-us/download/details.aspx?id=54765


## Image Randomization and Preprocessing

For this project, the images of cats and dogs were randomly selected and allocated into 3 seperate folders (training, test and validation). Contents in each folder are described below:
  - Training Set: 1000 images
  - Test Set: 200 images 
  - Validation Set: 150 images
 
**Note:** There are an equal number of cat and dog images in each set 
 
Following the allocation of the images, they were pre-processed using Tensorflow's ``ImageDataGenerator`` with the preprocessing argument set to the VGG16 preprocessing input. In addition, the images were put into batches of size 10 and also resized to a target size of 224 x 224. 

## Modeling 

Initially, a very common CNN with max pooling layers was used, and although it showed high accuracy on the training set, I noticed there was a noticeably big discrepency between the training and validation accuracies (>30%). As such, the original CNN was modified with a drop out layer and a L2 regularizer was used in hopes of decreasing this difference. Unfortunately, the modified network did not have much success in closing the gap between the 2 sets. Moving from this, I decided to apply transfer learning to the images to see if the results would be better; the pre-trained model which was used for this is the MobileNet V2 model. Both the modified CNN and mobilenet model can be seen below: 

  **Modified Model**
```
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same", input_shape= (224,224,3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2), 
    
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2), 
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2(0.0001))
    ])
```
  **Tranfer Model**
```
mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2()
mobilenet_model.trainable = False

model2 = tf.keras.Sequential([
    mobilenet_model,
    tf.keras.layers.Dense(2, activation = "softmax")
    ])
```
## Model Results 

**CNN Model** <br/>
Train:
  - Accuracy- 0.890
  - Loss- 0.926
  
 Validation: 
  - Accuracy- 0.505
  - Loss- 1.139
 
Test: 
  - Accuracy- 0.553
  - Loss 0.926

<img src="imgs/cnn model.png"  width = 400/>

**MobileNet Transfer Model** 

Train: 
  - Accuracy- 0.878
  - Loss 0.63

Validation: 
  - Accuracy- 0.980
  - Loss- 0.5362

Test: 
  - Accuracy- 0.940
  - Loss- 0.5402
  
<img src="imgs/transfer model.png"  width = 400/>

















