import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import subprocess

sys.path.append('../')
from convert_weights import h5_to_txt
from convert_weights import txt_to_h5

import numpy as np
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/home/jott1/Projects/CBRAIN-CAM/notebooks/tbeucler_devlog/hp_opt_conservation/SherpaResults/32col_random/normal_mse/Models/')
args =  parser.parse_args()

sample_input = np.array([[
    5, 4, 5, 9, 7, 5, 0, 3, 9, 4, 8, 0, 4, 5, 5, 4, 1, 0, 0, 4, 2, 8,
    2, 1, 7, 1, 6, 7, 4, 1, 4, 2, 6, 1, 9, 1, 7, 8, 7, 5, 8, 6, 3, 6,
    4, 7, 0, 5, 4, 1, 0, 1, 9, 6, 7, 3, 0, 3, 4, 1, 6, 2, 4, 1, 3, 7,
    6, 7, 8, 6, 7, 4, 5, 8, 8, 6, 0, 6, 9, 2, 5, 4, 1, 6, 9, 8, 7, 8,
    5, 1, 2, 1, 1, 6
]])

for file_name in os.listdir(args.model_dir):
    if not file_name.endswith('.h5'): continue

    # set appropriate paths
    model_path  = args.model_dir + file_name
    output_path = args.model_dir+file_name.replace('.h5', '.txt')

    print('Model path:', model_path)
    print('Output path:', output_path)

    # load keras model from h5
    try:
        model = load_model(model_path)
    except:
        print('Failed...')
        continue

    keras_predictions = model.predict(sample_input)[0]

    # convert h5 file to txt
    h5_to_txt(
        weights_file_name=model_path,
        output_file_name=output_path
    )

    cmd = ['../../build/bin/./test_bulk', output_path]

    # print('Keras predictions:', keras_predictions)

    # run
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    fortran_predictions = np.array(
        [float(num) for num in result.strip().split()],
        dtype=np.float32
    )
    # print('Fortran predictions:', fortran_predictions)

    # keras predictions must match neural fortran predictions
    assert np.allclose(keras_predictions, fortran_predictions, atol=1e-3)
