# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras import models
from keras import layers
import numpy as np

train_data=np.array([0.5,1.0,2.0,2.5,3.0])
train_label=np.array([2.5,-3.5,4.8,1.5,-2.0])

model = models.Sequential()

model.add(layers.Dense(4,activation='tanh',input_shape=(1,)))
model.add(layers.Dense(1,activation='linear'))

model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])

model.fit(train_data, train_label, epochs=1000, batch_size=5)

test_loss, test_acc = model.evaluate(train_data, train_label)

print('test_acc:', test_acc)
print('test_loss:',test_loss)

model.save('uyg1.h5')