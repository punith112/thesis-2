#IMPORTS
import os
import re
from random import randrange
import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
print('Keras image data format: {}'.format(K.image_data_format()))
import numpy as np
import matplotlib.pyplot as plt # (optional) for plotting and showing images inline

#CONSTANTS
MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history

#BUILDING THE NEURAL NETWORK

model = Sequential()
# input: 54x54 images with 1 channel -> (54, 54, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), input_shape=(54, 54, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64)) #fully-connected
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

batch_size = 16

training_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=10)

training_generator = training_datagen.flow_from_directory(
    'training_data',
    target_size=(54, 54),
    batch_size=batch_size,
    color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    'validation_data',
    target_size=(54, 54),
    batch_size=batch_size,
    color_mode='grayscale'
)

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=1200 // batch_size,
    epochs=20,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=160 // batch_size,
    workers=8,
)
model.save(MODEL_FILE)

x_batch, y_batch = next(training_generator)
print('Training data shape : ', x_batch.shape, y_batch.shape)
'''
plt.imshow(x_batch[0,:,:], cmap=plt.get_cmap('gray'))
plt.show()
'''
weight_conv2d_1 = model.layers[0].get_weights()[0][:,:,0,:]

col_size = 8
row_size = 4
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
for row in range(0,row_size):
  for col in range(0,col_size):
    ax[row][col].imshow(weight_conv2d_1[:,:,filter_index],cmap="gray")
    filter_index += 1
plt.show()

#arr_train_x_54x54 = np.reshape(x_batch.values, (x_batch.values.shape[0], 54, 54, 1))

test_index = randrange(x_batch.shape[0])
test_img = x_batch[test_index]
plt.imshow(test_img.reshape(54,54), cmap='gray')
plt.title("Index:[{}] Value:{}".format(test_index, y_batch[test_index]))
plt.show()
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_img.reshape(1,54,54,1))

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
      for col in range(0,col_size):
        ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
        activation_index += 1
    plt.show()

display_activation(activations, 8, 4, 1)

display_activation(activations, 8, 4, 2)

act_dense_3  = activations[14]

print(activations[14][0])

y = act_dense_3[0]
x = range(len(y))
plt.xticks(x)
plt.bar(x, y)
plt.show()
