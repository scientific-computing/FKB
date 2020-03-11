import sys
sys.path.append('../')
from convert_weights import h5_to_txt

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import RMSprop

weights_file_name = 'mnist_example.h5'
txt_file_name     = weights_file_name.replace('h5', 'txt')

batch_size = 128; num_classes = 10; epochs = 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255; x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# SEQUENTIAL MODEL
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# FUNCTIONAL MODEL
# input = x = Input(shape=(784,))
# x = Dense(512, activation='relu')(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(num_classes, activation='softmax')(x)
#
# model = Model(inputs=input, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

model.save(weights_file_name)

h5_to_txt(weights_file_name, txt_file_name)
