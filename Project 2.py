#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2


# In[3]:


mnist = tf.keras.datasets.mnist


# In[5]:


(x_train, y_train), (x_test, y_test)= mnist.load_data()


# In[6]:


x_train.shape


# In[8]:


plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap=plt.cm.binary)


# In[9]:


print (x_train[0])


# In[10]:


x_train = tf.keras.utils.normalize (x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap = plt.cm.binary)


# In[11]:


print(x_train[0])


# In[12]:


print(y_train[0])


# In[14]:


IMG_SIZE=28
x_trainr= np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr=np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples Dimensions", x_trainr.shape)
print("Testing Samples Dimension", x_testr.shape)


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[22]:


###Creating a Neural Network
model= Sequential()

### First Convolution Layer
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

### Second Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


### Third Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

##Fully Connected Layer 1
model.add (Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

###Fully Connected Layer
model.add(Dense(32))
model.add(Activation("relu"))

##Last Fully connected layer
model.add(Dense(10))
model.add(Activation('softmax'))


# In[23]:


model.summary()


# In[24]:


print ("Total Training Samples = ",len(x_trainr))


# In[25]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[26]:


model.fit(x_trainr, y_train, epochs=5, validation_split = 0.3)


# In[27]:


test_loss, test_acc=model.evaluate(x_testr, y_test)
print("Test loss on 10,000 test samples",test_loss)
print("Validation Accuracy on 10,000 test samples", test_acc)


# In[28]:


predicions = model.predict([x_testr])


# In[29]:


print (predicions)


# In[30]:


print (np.argmax(predicions[0]))


# In[31]:


plt.imshow(x_test[0])


# In[32]:


print (np.argmax(predicions[128]))


# In[33]:


plt.imshow(x_test[128])


# In[39]:


import cv2


# In[45]:


img = cv2.imread("C:/Users/DELL\OneDrive/Desktop/Handwritten Digit/digit7.png")


# In[46]:


print(img)


# In[47]:


plt.imshow(img)


# In[48]:


img.shape


# In[49]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[50]:


gray.shape


# In[51]:


resized = cv2.resize(gray, (28,28),interpolation = cv2.INTER_AREA)


# In[52]:


resized.shape


# In[53]:


newimg = tf.keras.utils.normalize (resized,axis=1)


# In[54]:


newimg=np.array(newimg).reshape(-1,IMG_SIZE, IMG_SIZE, 1)


# In[55]:


newimg.shape


# In[56]:


predicions = model.predict(newimg)


# In[57]:


print (np.argmax(predicions))


# In[ ]:




