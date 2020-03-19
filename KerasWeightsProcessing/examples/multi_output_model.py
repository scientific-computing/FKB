import os
import sys
sys.path.append('../')
import numpy as np
import convert_weights
import tensorflow as tf

############## REPRODUCIBILITY ############
tf.set_random_seed(0)
np.random.seed(0)
###########################################

from keras.models import load_model
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, BatchNormalization, Input

input = x = Input((5,))
for i in range(3):
    x = Dense(30)(x)
    x = BatchNormalization()(x)

# MULTIPLE OUTPUTS
output1 = Dense(1)(x)
output2 = Dense(1)(x)
output3 = Dense(1)(x)

# CREATE THE MODEL
multi_output_model = Model(input,outputs=[output1, output2, output3])

multi_output_model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
# SAVE TO FILE FOR PARSING
multi_output_model.save('multi_output_model.h5')

# CONVERT TO TXT
convert_weights.h5_to_txt('multi_output_model.h5', 'single_output_model.txt')
# CONVERT TO H5
convert_weights.txt_to_h5('single_output_model.txt', 'single_output_model.h5')

single_output_model = load_model('single_output_model.h5')

# GRAPHIC PLOT OF MODEL
plot_model(multi_output_model, to_file='../../Figures/multi_output_model.png', show_shapes=True, show_layer_names=True)
plot_model(single_output_model, to_file='../../Figures/single_output_model.png', show_shapes=True, show_layer_names=True)

# TEST INPUT
input = np.array(
    [[1,2,3,4,5]]
)

# COMPARE PREDICTIONS FROM MULTI OUTPUT AND SINGLE OUTPUT MODELS
multiple_output = np.array(multi_output_model.predict(input)).squeeze()
single_output   = single_output_model.predict(input).squeeze()

assert np.allclose(multiple_output, single_output)

print('MULTI-OUTPUT:', multiple_output)
print('SINGLE-OUTPUT:', single_output)
