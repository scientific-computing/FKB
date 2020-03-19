import keras
import argparse
import numpy as np
import subprocess
import os

import sys
sys.path.append('../')
from convert_weights import h5_to_txt
from convert_weights import txt_to_h5

# set random seeds for reproducibility
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LeakyReLU, Dropout, BatchNormalization
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--num_dense_layers', default=5, type=int)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'linear', 'leakyrelu'])
parser.add_argument('--model_type', default='sequential', type=str, choices=['sequential', 'functional'])
args =  parser.parse_args()

# construct weights file name based on settings
weights_file = '{model_type}_{num_dense_layers}_{activation}_{dropout}_{batchnorm}.h5'.format(
    model_type=args.model_type,
    num_dense_layers=args.num_dense_layers,
    activation=args.activation,
    dropout=args.dropout,
    batchnorm=args.batchnorm
)

if args.activation == 'leakyrelu':
    activation = 'linear'
else:
    activation = args.activation

if args.model_type == 'sequential':
    model = Sequential()

    for i in range(args.num_dense_layers - 1):
        if i == 0:
            # first layer requires input shape
            model.add(Dense(32, input_shape=(5,), activation=activation))
        else:
            model.add(Dense(32, activation=activation))

        # leakyrelu is an advanced activation
        # requires its own "layer"
        if args.activation == 'leakyrelu':
            model.add(LeakyReLU(alpha=0.3))

        # add dropout layer if required
        if args.dropout != 0:
            model.add(Dropout(args.dropout))

        if args.batchnorm:
            model.add(BatchNormalization())

    # output layer with 2 nodes
    model.add(Dense(2))
else:
    input = x = Input((5,))

    for i in range(args.num_dense_layers - 1):
        x = Dense(32, activation=activation)(x)

        if args.activation == 'leakyrelu':
            x = LeakyReLU(alpha=0.3)(x)

        if args.dropout != 0:
            x = Dropout(args.dropout)(x)

        if args.batchnorm:
            x = BatchNormalization()(x)

    x = Dense(2)(x)

    model = Model(inputs=input, outputs=x)

# compile model
model.compile(loss='mse', optimizer='sgd')

example_input = np.array(
    [[1,2,3,4,5]]
)

if args.train:
    example_input = np.array([
        [1,2,3,4,5],
        [0,1,2,3,4],
        [-1,0,1,2,3]
    ])
    example_target = np.array([
        [1,2],
        [0,1],
        [-1,0]
    ])

    model.fit(
        example_input,example_target,
        epochs=5
    )

keras_predictions = model.predict(example_input)[0]

# save the weights
model.save(weights_file)
# convert h5 file to txt
h5_to_txt(
    weights_file_name=weights_file,
    output_file_name=None
)
txt_to_h5(
    weights_file_name=weights_file.replace('.h5','.txt'),
    output_file_name=None
)

cmd = ['../../build/bin/./test_keras', weights_file.replace('.h5', '.txt')]

print('Keras predictions:      ', keras_predictions)

if os.path.exists(weights_file.replace('.h5','_converted.h5')):
    model2 = load_model(weights_file.replace('.h5','_converted.h5'))
    keras_predictions2 = model2.predict(example_input)[0]
    print('Other Keras predictions:', keras_predictions2)

# run
result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
fortran_predictions = np.array(
    [float(num) for num in result.strip().split()],
    dtype=np.float32
)
print('Fortran predictions:    ', fortran_predictions)

# keras predictions must match neural fortran predictions
assert np.allclose(keras_predictions, keras_predictions2,fortran_predictions)
