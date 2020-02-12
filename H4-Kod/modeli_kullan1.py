# -*- coding: utf-8 -*-
"""
mnist ile eğitilmiş modeli 
test görüntüleri ilekullanan program
"""
from keras.datasets import mnist
from keras import models
import numpy as np
from matplotlib import pyplot as plt

#modeli yükle
model=models.load_model('modelmnist.h5')
#test görüntülerini yükle
(x_train, train_labels),(x_test, test_labels)=\
mnist.load_data()


# test görüntüsü  111. görüntü
giris=x_test[111,:,:]/255

plt.imshow(giris)

#Eğitilen modele uygula
giris=giris.reshape(1,28*28)
y=model.predict(giris)

#sonucu göster
rakam=np.argmax(y)
print('rakam=',rakam)