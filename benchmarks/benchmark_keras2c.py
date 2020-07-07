import pickle
import sys
import os
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
import subprocess
import time
from keras2c.keras2c_main import k2c

num_cores = 1
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 0})
session = tf.Session(config=config)
K.set_session(session)


def build_and_run(name, return_output=False, cc='gcc'):

    cwd = os.getcwd()
    os.chdir(os.path.abspath('./include/'))
    lib_code = subprocess.run(['make', 'CC={}'.format(cc)]).returncode
    os.chdir(os.path.abspath(cwd))
    if lib_code != 0:
        return 'lib build failed'

    ccflags = ' -O3 -std=c99 -I./include/'
    if cc == 'gcc':
        ccflags += ' -march=native'
    elif cc == 'icc':
        ccflags += ' -xHost'

    comp = cc + ccflags + ' -o ' + name + ' ' + name + '.c ' + \
        name + '_test_suite.c -L./include/ -l:libkeras2c.a -lm'
    build_code = subprocess.run(comp.split()).returncode
    if build_code != 0:
        return 'build failed'

    if return_output:
        proc_output = subprocess.run(
            ['./' + name], capture_output=True, text=True)
    else:
        proc_output = subprocess.run(['./' + name])
    rcode = proc_output.returncode
    if rcode == 0:
        if not os.environ.get('CI'):
            subprocess.run('rm ' + name + '*', shell=True)
            return (rcode, proc_output.stdout) if return_output else rcode
    return rcode


def time_model(model, num_tests, num_runs):
    pytimes = np.zeros(num_runs)
    ctimes = np.zeros(num_runs)
    nparams = model.count_params()

    k2c(model, 'foo', num_tests=num_tests, malloc=False, verbose=False)
    out = build_and_run('foo', True, 'icc')[1]

    for j in range(num_runs):
        ctimes[j] = float(out.split('\n')[0].split(' ')[-3])

    inp = np.random.random((num_tests, *model.input_shape[1:]))
    inp = np.expand_dims(inp, 1)
    for j in range(num_runs):
        t0 = time.time_ns()
        for i in range(num_tests):
            _ = model.predict(inp[i])
        t1 = time.time_ns()
        pytimes[j] = (t1-t0)/10**9/num_tests
    return nparams, ctimes, pytimes


time_data = {}


"""Dense Model"""
size = []
ctimes = []
pytimes = []
save_dims = []
save_layers = []

num_tests = 10
num_runs = 10
nlayers = [1, 2, 4]
dims = [8, 16, 32]

for nl in nlayers:
    for dim in dims:
        inshp = (dim,)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            dim, input_shape=inshp, activation='relu'))
        if nl > 1:
            for i in range(nl-1):
                model.add(keras.layers.Dense(dim, activation='relu'))
        model.build()
        nparams, ctime, pytime = time_model(model, num_tests, num_runs)
        size.append(nparams)
        ctimes.append(ctime)
        pytimes.append(pytime)
        save_dims.append(dim)
        save_layers.append(nl)

time_data['Fully Connected'] = {'size': size,
                                'layers': save_layers,
                                'dim': save_dims,
                                'ctimes': ctimes,
                                'pytimes': pytimes}

with open('k2c_benchmark_times.pkl', 'wb+') as f:
    pickle.dump(time_data, f)


"""Conv1D Model"""
size = []
ctimes = []
pytimes = []
save_dims = []
save_layers = []

num_tests = 10
num_runs = 10
nlayers = [1, 2, 4]
dims = [8, 16, 32]


for nl in nlayers:
    for dim in dims:
        inshp = (dim, 4)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(4, kernel_size=int(
            dim**.5), input_shape=inshp, padding='same'))
        if nl > 1:
            for i in range(nl-1):
                model.add(keras.layers.Conv1D(
                    10 + 2*i, kernel_size=int(dim**.5), padding='same'))

        model.build()
        nparams, ctime, pytime = time_model(model, num_tests, num_runs)
        size.append(nparams)
        ctimes.append(ctime)
        pytimes.append(pytime)
        save_dims.append(dim)
        save_layers.append(nl)

time_data['Conv1D'] = {'size': size,
                       'layers': save_layers,
                       'dim': save_dims,
                       'ctimes': ctimes,
                       'pytimes': pytimes}


with open('k2c_benchmark_times.pkl', 'wb+') as f:
    pickle.dump(time_data, f)


"""Conv2D Model"""
size = []
ctimes = []
pytimes = []
save_dims = []
save_layers = []

num_tests = 10
num_runs = 10
nlayers = [1, 2, 4]
dims = [8, 12, 16, 24, 32]


for nl in nlayers:
    for dim in dims:
        inshp = (dim, dim, 3)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(5, kernel_size=int(
            np.log2(dim)), input_shape=inshp, padding='same'))
        if nl > 1:
            for i in range(nl-1):
                model.add(keras.layers.Conv2D(
                    10+2*i**2, kernel_size=int(np.log2(dim)), padding='same'))

        model.build()
        nparams, ctime, pytime = time_model(model, num_tests, num_runs)
        size.append(nparams)
        ctimes.append(ctime)
        pytimes.append(pytime)
        save_dims.append(dim)
        save_layers.append(nl)

time_data['Conv2D'] = {'size': size,
                       'layers': save_layers,
                       'dim': save_dims,
                       'ctimes': ctimes,
                       'pytimes': pytimes}


with open('k2c_benchmark_times.pkl', 'wb+') as f:
    pickle.dump(time_data, f)


"""LSTM Model"""
size = []
ctimes = []
pytimes = []
save_dims = []
save_layers = []

num_tests = 10
num_runs = 10
nlayers = [1, 2, 4]
dims = [8, 16, 32]


for nl in nlayers:
    for dim in dims:
        inshp = (int(np.sqrt(dim)), dim)
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(
            dim, return_sequences=True, input_shape=inshp))
        if nl > 1:
            for i in range(nl-1):
                model.add(keras.layers.LSTM(
                    dim, return_sequences=True, input_shape=inshp))

        model.build()
        nparams, ctime, pytime = time_model(model, num_tests, num_runs)
        size.append(nparams)
        ctimes.append(ctime)
        pytimes.append(pytime)
        save_dims.append(dim)
        save_layers.append(nl)

time_data['LSTM'] = {'size': size,
                     'layers': save_layers,
                     'dim': save_dims,
                     'ctimes': ctimes,
                     'pytimes': pytimes}


with open('k2c_benchmark_times.pkl', 'wb+') as f:
    pickle.dump(time_data, f)
