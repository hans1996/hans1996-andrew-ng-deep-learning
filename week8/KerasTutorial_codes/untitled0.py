# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:12:50 2018

@author: user
"""
# In[]
#from keras.models import load_model


# 載入模型
#model = load_model('model.h5')
# In[]
#happyModel.summary()
# In[]
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
# In[]
model = VGG16(weights='imagenet', include_top=True) 
# In[]
### START CODE HERE ###
img_path = 'C:/Users/user.LAPTOP-N880RBTF/Desktop/deep learning/week8/KerasTutorial_codes/images/my_image.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# In[]
print(happyModel.predict(x))
# In[]
plot_model(model, to_file='Model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
# In[]
# In[1]:


import numpy as np
#import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# get_ipython().run_line_magic('matplotlib', 'inline')
# In[]
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
# In[]
import keras
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"] )
# In[]
preds =  model.evaluate(X_test, Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
# In[]
model.fit(x = X_train, y = Y_train,epochs=5 ,batch_size=32) 