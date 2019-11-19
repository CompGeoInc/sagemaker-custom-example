import os
import shutil
from os import path

import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import csv
import json
import traceback


SM_DOCKER_REFERENCE = 'https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html'

BASE_PATH = '/opt/ml/'
CONFIG_PATH = path.join(BASE_PATH, 'input/config')
PARAMETERS_FILE = path.join(CONFIG_PATH, 'hyperparameters.json')
INPUT_CONFIG_FILE = path.join(CONFIG_PATH, 'inputdataconfig.json')

MODEL_PATH = path.join(BASE_PATH, 'model')
TRAINED_MODEL_DIR = path.join(MODEL_PATH, 'trained')

INPUT_DATA_PATH = path.join(BASE_PATH, 'input/data')
TRAINING_DATA_FILE = path.join(INPUT_DATA_PATH, 'train/examples.csv')
TEST_DATA_FILE = path.join(INPUT_DATA_PATH, 'test/examples.csv')

OUTPUT_PATH = path.join(BASE_PATH, 'output')
FAILURE_FILE = path.join(OUTPUT_PATH, 'failure')

exit_code = 0

try:
    assert len(sys.argv) >= 2, 'SageMaker job argument (train, serve, etc.) expected, see ' + SM_DOCKER_REFERENCE

    command = sys.argv[1]

    if command == 'train':
        # get hyperparameters
        with open(PARAMETERS_FILE) as params_file:
            parameters = json.load(params_file)

        # get input data config (just an example - not needed yet)
        with open(INPUT_CONFIG_FILE) as input_config_file:
            input_config = json.load(input_config_file)

        # load training data
        xs_data = []
        ys_data = []
        with open(TRAINING_DATA_FILE) as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                ys_data.append(float(row[0]))
                xs_data.append(float(row[1]))

        model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        xs = np.array(xs_data, dtype=float)
        ys = np.array(ys_data, dtype=float)
        model.fit(xs, ys, epochs=int(parameters['epochs']))

        # save trained model
        try:
            os.makedirs(TRAINED_MODEL_DIR)
        except FileExistsError:
            shutil.rmtree(TRAINED_MODEL_DIR)
        tf.saved_model.save(model, TRAINED_MODEL_DIR)
        print('Model successfully saved to', TRAINED_MODEL_DIR)

    # this is not a real SageMaker command, it's just here to test our model locally
    elif command == 'test':
        xs_data = []
        with open(TEST_DATA_FILE) as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                xs_data.append(float(row[0]))
        model = tf.saved_model.load(TRAINED_MODEL_DIR)
        infer = model.signatures["serving_default"]
        xs = tf.constant(xs_data, dtype=tf.float32, shape=(1, len(xs_data)))
        ys = infer(xs)['dense'].numpy()
        with np.printoptions(precision=3):
            print('Prediction output:', ys[0])
    else:
        raise RuntimeError('I don\'t know how to "{}", see {}'.format(command, SM_DOCKER_REFERENCE))

# write exception to failure file
except Exception as e:  # pylint: disable=broad-except
    failure_msg = "error: \n%s\n%s" % (traceback.format_exc(), str(e))
    print(failure_msg)
    with open(FAILURE_FILE, 'w') as f:
        f.write(failure_msg)
    exit_code = 1

exit(exit_code)
