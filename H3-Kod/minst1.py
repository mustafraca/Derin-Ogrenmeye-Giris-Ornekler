# -*- coding: utf-8 -*-
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

#1- Veri setini yükle
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

#2-Verilerin hazırlanması
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#3- Eğitilecek ağ yapısını tanımla
network = models.Sequential()

#Dense layer
network.add(layers.Dense(16,activation='relu',input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))




# Derle
network.compile(
        optimizer='rmsprop', #adam, sgd
        loss='categorical_crossentropy',#loss='mean_squared_error', 
        metrics=['accuracy'])#
#network.compile(
#        optimizer='adam', #adam, sgd
#        loss='mean_squared_error',#loss='mean_squared_error', 
#        metrics=['accuracy'])
#
#Eğit
history=network.fit(train_images, train_labels, epochs=5, batch_size=256)

#Eğitimde kullanılmamış verilerle başarımı test et
test_loss, test_acc = network.evaluate(
        test_images, 
        test_labels)

print('test_acc:', test_acc)
print('test_loss:',test_loss)

#eğitilmiş modeli ve ağırlıkları kaydet
network.save('mnist.h5')
#network.save_weights('mnist.w1')

#------------------------------------------------------------------------------
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

 
