from keras.datasets import reuters
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

max_features = 1000
maxlen = 100


print('Loading data...')

(input_train, y_train), (input_test, y_test) = reuters.load_data(
        num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from keras.layers import Dense

model = Sequential()

model.add(Embedding(max_features, 32))

model.add(SimpleRNN(32))

model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)

