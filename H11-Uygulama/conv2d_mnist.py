# -*- coding: utf-8 -*-
"""
Training the convnet on MNIST images

"""


from keras import layers
from keras import models
import matplotlib.pyplot as plt

model = models.Sequential()

model.add(layers.Conv2D(8,(3, 3),
                        use_bias=False,
                        strides=(2,2),
                        padding='same',
                        activation='relu',input_shape=(28,28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(8, 
                        (5, 5), 
                        padding='same',
                        activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


model.summary()

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) =\
 mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) 
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from keras import losses
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', #losses.sparse_categorical_crossentropy
              metrics=['accuracy'])

history = model.fit(train_images, 
          train_labels, 
          epochs=5, 
          validation_split=0.1,
          batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc=",test_acc)
print('test_loss:',test_loss)

model.save('mnist_conv_model')

print(history.history.keys())

plt.figure(1)
plt.plot(history.history['acc'])
plt.title(' accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.title(' loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()