import h5py
import json
import warnings
import argparse
import numpy as np

import numpy as np
import math
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Input, Activation
from keras import optimizers

INPUT = ['input']
ACTIVATIONS = ['relu', 'linear', 'leakyrelu', 'sigmoid']
SUPPORTED_LAYERS = ['dense', 'dropout', 'batchnormalization'] + ACTIVATIONS + INPUT

def txt_to_h5(weights_file_name, output_file_name=''):
    '''
    Convert a txt file to Keras h5 file

    REQUIRED:
        weights_file_name (str): path to a txt file used by neural fortran
    OPTIONAL:
        output_file_name  (str): desired output path for the produced h5 file
    '''

    lr               = False
    bias             = []                                       # dense layer
    weights          = []                                       # dense layer
    batchnorm_params = []                                       # batchnormalization layers

    bias_count       = 0
    weights_count    = 0
    batchnorm_count  = 0

    with open(weights_file_name, mode='r') as weights_file:
        lines = weights_file.readlines()

        for idx, line in enumerate(lines):

            if idx == 0:
                num_layers = int(line)
                continue

            line = line.strip().split("\t")
            layer_type = line[0]

            if layer_type in SUPPORTED_LAYERS:
                param = line[1]
                if layer_type == 'input':
                    input = Input(shape=(int(param),), name = "input")
                    x     = input

                elif layer_type == 'dense':
                    bias_count += 1; weights_count += 1
                    x = Dense(int(param),name = "dense_{}".format(weights_count))(x)

                elif layer_type == 'dropout':
                    x = Dropout(float(param))(x)

                elif layer_type == 'relu':
                    x = Activation('relu')(x)

                elif layer_type == 'relu':
                    x = Activation('relu',alpha=float(param))(x)

                elif layer_type == 'batchnormalization':
                    batchnorm_count += 4
                    x = BatchNormalization(name='batch_normalization_{}'.format(batchnorm_count // 4))(x)
                elif layer_type == 'linear':
                    x = Activation('linear')(x)

            elif not layer_type.isalpha():
                if lr == False:
                    lr = float(line[0]); continue

                # found bias or weights numbers
                w = np.asarray([float(num) for num in line])

                if bias_count > 0:
                    bias_count -= 1
                    bias.append(w)

                elif weights_count > 0:
                    weights_count -= 1
                    weights.append(w)

                elif batchnorm_count > 0:
                    batchnorm_count -= 1
                    batchnorm_params.append(w)

    # create model
    model = Model(inputs=input, outputs=x)

    # compile model
    model.compile(
        loss='mse',
        optimizer=optimizers.SGD(lr),
        metrics=['accuracy']
    )

    # set weights and biases
    for idx, w in enumerate(weights):
        name    = 'dense_{}'.format(idx+1)
        layer   = model.get_layer(name)
        w       = w.reshape(layer.output_shape[1],layer.input_shape[1]).T
        layer.set_weights( [w, bias[idx]] )

    # set batchnorm parameters
    for idx in range(0,len(batchnorm_params),4):
        params  = batchnorm_params[idx:idx+4]
        name    = 'batch_normalization_{}'.format(idx // 4 + 1)
        layer   = model.get_layer(name)
        layer.set_weights([
            params[1],
            params[0],
            params[2],
            params[3],
        ])

    # view summary

    if not output_file_name:
        # if not specified will use path of weights_file with h5 extension
        output_file_name = weights_file_name.replace('.txt', '_converted.h5')

    model.save(output_file_name)

def h5_to_txt(weights_file_name, output_file_name=''):
    '''
    Convert a Keras h5 file to a txt file

    REQUIRED:
        weights_file_name (str): path to a Keras h5 file
    OPTIONAL:
        output_file_name  (str): desired path for the produced txt file
    '''

    info_str         = '{name}\t{info}\n'                       # to store in layer info; config of network
    bias             = []                                       # dense layer
    weights          = []                                       # dense layer
    layer_info       = []                                       # all layers
    batchnorm_params = []                                       # batchnormalization layers
    out_bias_dict    = {}                                       # output bias if multiple dense outputs
    out_weights_dict = {}                                       # output weights if multiple dense outputs
    out_bias         = []                                       # output bias if multiple dense outputs
    out_weights      = []                                       # output weights if multiple dense outputs
    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        # weights of model
        model_weights = weights_file['model_weights']
        keras_version = weights_file.attrs['keras_version']

        if 'training_config' in weights_file.attrs:
            training_config = weights_file.attrs['training_config'].decode('utf-8')
            training_config = training_config.replace('true','True')
            training_config = training_config.replace('false','False')
            training_config = training_config.replace('null','None')
            training_config = eval(training_config)

            if 'learning_rate' in training_config['optimizer_config']['config']: learning_rate = training_config['optimizer_config']['config']['learning_rate']
            else: learning_rate = training_config['optimizer_config']['config']['lr']
        else:
            warnings.warn('Model has not been compiled: Setting learning rate default')
            learning_rate = 0.001

        # Decode using the utf-8 encoding; change values for eval
        model_config = weights_file.attrs['model_config'].decode('utf-8')
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')
        model_config = model_config.replace('null','None')
        # convert to dictionary
        model_config = eval(model_config)

        # store first dimension for the input layer
        layer_info.append(
            info_str.format(
                name = 'input',
                info = model_config['config']['layers'][0]['config']['batch_input_shape'][1]
            )
        )

        # check what type of keras model sequential or functional
        if model_config['class_name'] == 'Model':
            layer_config = model_config['config']['layers'][1:]

            # get names of input layers and output layers
            if model_config['config'].get('output_layers'):
                output_layers = model_config['config'].get('output_layers',[])
                output_names = [layer[0] for layer in output_layers]

            if model_config['config'].get('input_layers'):
                input_layers = model_config['config'].get('input_layers',[])
                input_names = [layer[0] for layer in input_layers]

        else:
            layer_config = model_config['config']['layers']

        for idx,layer in enumerate(layer_config):
            name       = layer['config']['name']
            class_name = layer['class_name'].lower()

            if class_name not in SUPPORTED_LAYERS:
                warning_str = 'Unsupported layer, %s, found! Skipping...' % class_name
                warnings.warn(warning_str)
                continue
            elif class_name == 'dense':
                # get weights and biases out of dictionary
                layer_weights = np.array(
                    model_weights[name][name]['kernel:0']
                )

                if 'bias:0' in model_weights[name][name]:
                    layer_bias = np.array(
                        model_weights[name][name]['bias:0']
                    )
                else:
                    warnings.warn('No bias found: Replacing with zeros')
                    layer_bias = np.zeros(layer_weights.shape[1])

                # store bias values
                bias.append(layer_bias)
                # store weight value
                weights.append(layer_weights)

                activation = layer['config']['activation']

                if activation not in ACTIVATIONS:
                    warning_str = 'Unsupported activation, %s, found! Replacing with Linear.' % activation
                    warnings.warn(warning_str)
                    activation = 'linear'

                # store dimension of hidden dim
                layer_info.append(
                    info_str.format(
                        name = class_name,
                        info = layer_weights.shape[1]
                    )
                )
                # add information about the activation
                layer_info.append(
                    info_str.format(
                        name = activation,
                        info = 0
                    )
                )
            elif class_name == 'batchnormalization':
                # get beta, gamma, moving_mean, moving_variance from dictionary
                for key in sorted(model_weights[name][name].keys()):
                    # store batchnorm params
                    batchnorm_params.append(
                        np.array(
                            model_weights[name][name][key]
                        )
                    )

                # store batchnorm layer info
                layer_info.append(
                    info_str.format(
                        name = class_name,
                        info = 0
                    )
                )

            elif class_name == 'dropout':
                layer_info.append(
                    info_str.format(
                        name = class_name,
                        info = layer['config']['rate']
                    )
                )

            elif class_name in ACTIVATIONS:
                # replace previous dense layer with the advanced activation function (LeakyReLU)
                layer_info[-1] = info_str.format(
                    name = class_name,
                    info = layer['config']['alpha']
                )

            # if there are multiple outputs, remove what was just added
            # and combine in single output
            if 'output_names' in locals() and len(output_names) > 1 and name in output_names:
                try:
                    if class_name not in ['dense']:
                        warnings.warn('Only multiple dense outputs allowed! Skipping...')
                        continue
                    # remove last bias and weights
                    # start building up the combined output bias and weights
                    out_bias_dict[name]     = bias.pop()
                    out_weights_dict[name]  = weights.pop()
                    layer_info, out_info    = layer_info[:-2], layer_info[-2:]
                except:
                    pass
    if 'output_names' in locals() and len(output_names) > 1:
        #need to combine outputs here into one layer
        out_info[0] = out_info[0].replace('1',str(len(output_names)))
        layer_info.extend(out_info)

        for name in output_names:
            out_bias.extend(out_bias_dict.get(name))
            out_weights.append(out_weights_dict.get(name))

        bias.append(
            np.array(out_bias).squeeze()
        )
        weights.append(
            np.array(out_weights).squeeze().T
        )

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(len(layer_info)) + '\n')

        output_file.write(
            ''.join(layer_info)
        )

        output_file.write(
            str(learning_rate) + '\n'
        )

        for b in bias:
            bias_str = '\t'.join(
                '{:0.7e}'.format(num) for num in b.tolist()
            )
            output_file.write(bias_str + '\n')

        for w in weights:
            weights_str = '\t'.join(
                '{:0.7e}'.format(num) for num in w.T.flatten()
            )
            output_file.write(weights_str + '\n')

        for b in batchnorm_params:
            param_str = '\t'.join(
                '{:0.7e}'.format(num) for num in b.tolist()
            )
            output_file.write(param_str + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", type=str, help='path to desired file to be processed')
    parser.add_argument('--output_file', default='', type=str)
    args =  parser.parse_args()

    if args.weights_file.endswith('.h5'):
        h5_to_txt(
            weights_file_name=args.weights_file,
            output_file_name=args.output_file
        )
    elif args.weights_file.endswith('.txt'):
        txt_to_h5(
            weights_file_name=args.weights_file,
            output_file_name=args.output_file
        )

    else:
        warnings.warn('Unsupported file extension')
