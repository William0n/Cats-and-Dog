import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# import os 
# import glob
# import shutil
# import random

import matplotlib.pyplot as plt 
import PIL



## creating different folders for training, test, and validation sets 

os.makedirs('train/dog')
os.makedirs('train/cat')
os.makedirs('valid/dog')
os.makedirs('valid/cat')
os.makedirs('test/dog')
os.makedirs('test/cat')
## iteraing through the 'Dog' image folder to randomly select images to
## alloacte into the specifed folders. Code below commented out 
## to avoid another shuffling of images when reruning the code.

#for c in random.sample(glob.glob('Dog/*'), 500):
    shutil.move(c, 'train/dog')
#for c in random.sample(glob.glob('Cat/*'), 500):
    shutil.move(c, 'train/cat')
#for c in random.sample(glob.glob('Dog/*'), 100):
    shutil.move(c, 'valid/dog')
#for c in random.sample(glob.glob('Cat/*'), 100):
    shutil.move(c, 'valid/cat')
#for c in random.sample(glob.glob('Dog/*'), 75):
    shutil.move(c, 'test/dog')
#for c in random.sample(glob.glob('Cat/*'), 75):
    shutil.move(c, 'test/cat')

train_path = 'cats and dogs\\train'
valid_path = 'cats and dogs\\valid'
test_path = 'cats and dogs\\test'

##### Using gray scale images 
train_batches = ImageDataGenerator(rescale =1./255) \
    .flow_from_directory(directory = train_path, target_size = (224,224),color_mode= "grayscale",
                         classes =['cat', 'dog'],
                         batch_size = 10 )
    
    
valid_batches = ImageDataGenerator(rescale =1./255) \
    .flow_from_directory(directory = valid_path, target_size = (224,224),color_mode= "grayscale",
                         classes =['cat', 'dog'],
                         batch_size = 10 )
    
test_batches = ImageDataGenerator(rescale =1./255) \
    .flow_from_directory(directory = test_path, target_size = (224,224),color_mode= "grayscale",
                         classes =['cat', 'dog'],
                         batch_size = 10 )    

    
## Using rgb color images with vgg16 preprocessing 

train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = train_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )
    
    
valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = valid_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )
    
# if you want vgg16 in grayscale put color_mode= 'grayscale' in the flowfrom directory

test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = test_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )

# To see the first 5 images

imgs, labels = next(test_batches)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
plotImages(imgs[:5])

### how to see a single grayscaled/rgb images below 

plt.imshow(np.squeeze(imgs[9]), cmap=plt.cm.binary)

## OPTIONAL implement early stop in sequential layering 
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss')

## Using two convolution layer with 3x3 kernel and max pooling layers  

model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same", input_shape= (224,224,3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2), 
    
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = "relu", padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2), 

# can do a 50% dropout layer first by adding tf.keras.layers.Dropout(0.5)#
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2(0.0001))
    ])

## compile uses categorial cross entropy instead of SPARSE cross entropy 
## because labels are one hot encoded
model.compile(loss = 'categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy'])

history_cnn = model.fit(train_batches, validation_data = valid_batches, epochs = 5)

test_result = model.evaluate(test_batches)

#### graphs for loss and accuracy 

plt.figure(figsize = (14,8))
plt.subplot(1,2,1)

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])

plt.subplot(1,2,2)
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])


plt.subplots_adjust(wspace = 0.25)
plt.show()

##### TRANSFER LEARNING USING MOBILENET V2


mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2()
mobilenet_model.summary()

train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input) \
    .flow_from_directory(directory = train_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )
    

valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input) \
    .flow_from_directory(directory = valid_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input) \
    .flow_from_directory(directory = test_path, target_size = (224,224),
                         classes =['cat', 'dog'],
                         batch_size = 10 )

mobilenet_model.trainable = False

#### code to make model without last 2 layers of pretrained model
x = mobilenet_model.layers[-2].output
output= tf.keras.layers.Dense(2, activation = 'softmax')(x)

model2 = tf.keras.Model(inputs = mobilenet_model.input, outputs = output)

### Using the pretrained mobilenet model
    
model2 = tf.keras.Sequential([
    mobilenet_model,
    tf.keras.layers.Dense(2, activation = "softmax")
    ])


model2.compile(loss = 'categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy'])
history = model2.fit(train_batches, validation_data = valid_batches, epochs = 5)

result = model2.evaluate(test_batches)

plt.figure(figsize = (14,8))
plt.subplot(1,2,1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'])


plt.subplots_adjust(wspace = 0.25)
plt.show()

#model.save('cats and dogs/ pretrained_cd_model.h5')
 model.predict(imgs[[1]])

#loaded_model = tf.keras.models.load_model('cats and dogs/ pretrained_cd_model.h5')

