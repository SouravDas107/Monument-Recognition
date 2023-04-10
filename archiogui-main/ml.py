from google.colab import drive
drive.mount('/content/gdrive')

!pip install imageio

from pathlib import Path
import os
import matplotlib.pyplot as plt
import cv2
import glob
import tensorflow as tf
from imageio import imread
from tqdm import tqdm
from PIL import Image
from numpy import*

train_path = Path("/content/gdrive/My Drive/Data/Train")
val_path = Path("/content/gdrive/My Drive/Data/Test")

data_list = os.listdir(train_path)
data_list

IMAGE_SIZE = [250, 250]

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense

x = Flatten()(vgg.output)

prediction = Dense(len(data_list), activation='softmax')(x)

from keras.models import Model
model = Model(inputs=vgg.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
            rotation_range=9, # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.2,  # randomly shift images horizontally 
            height_shift_range=0.2)  # randomly shift images vertically 

val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/gdrive/My Drive/Data/Train',
                                                 target_size = (250, 250),
                                                 batch_size = 32,   
                                                 class_mode = 'categorical')

test_set = val_datagen.flow_from_directory('/content/gdrive/My Drive/Data/Test',
                                            target_size = (250, 250),
                                            batch_size = 32,
                                            class_mode = 'categorical')

fit_=model.fit_generator(training_set, validation_data=test_set, epochs=15)
model.save('my_model.h1')

import numpy as np
import keras.utils as image
#from keras.preprocessing import image
img = image.load_img("/content/gdrive/My Drive/Dataset/vic.jpg",target_size=(250,250))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

output = saved_model.predict(img)

print(output[0])

