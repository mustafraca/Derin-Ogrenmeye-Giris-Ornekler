# -*- coding: utf-8 -*-
"""
sınıflandırma modeli tanımlanması ve 
mnist veri seti ile eğitimi
"""

from keras.datasets import mnist
(x_train, train_labels),(x_test, test_labels)=\
mnist.load_data()

x_train=x_train.reshape(60000,28*28)
x_train=x_train.astype('float32')/255

x_test=x_test.reshape(10000,28*28)
x_test=x_test.astype('float32')/255

from keras import models
from keras import layers
from keras import activations
model=models.Sequential()

model.add(layers.Dense(256,
                       activation=activations.relu,
                       input_shape=(28*28,)))
model.add(layers.Dense(10,
                       activation='softmax'))
#eğitim parametreleri
from keras import optimizers
from keras import losses
model.compile(optimizer='rmsprop',
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#modeli eğit
model.fit(x_train,
          train_labels,
          epochs=5,
          batch_size=100)
#modeli test et
test_loss,test_acc=\
model.evaluate(x_test,test_labels)

print('test_acc=',test_acc)

model.save('modelmnist.h5')




