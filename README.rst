#######
keras2c
#######

|Build-Status| |Codecov|

|License| |DOI|


keras2c is a library for deploying keras neural networks in C99, using only standard libraries.
It is designed to be as simple as possible for real time applications.

Quickstart
**********

After cloning the repo, install the necessary packages with ``pip install -r requirements.txt``.

keras2c can be used from the command line:

.. code-block:: bash

    python -m keras2c [-h] [-m] [-t] model_path function_name

    A library for converting the forward pass (inference) part of a keras model to
        a C function

    positional arguments:
      model_path         File path to saved keras .h5 model file
      function_name      What to name the resulting C function
     
    optional arguments:
      -h, --help         show this help message and exit
      -m, --malloc       Use dynamic memory for large arrays. Weights will be
                         saved to .csv files that will be loaded at runtime
      -t , --num_tests   Number of tests to generate. Default is 10


It can also be used with a python environment in the following manner:

.. code-block:: python

    from keras2c import k2c
    k2c(model, function_name, malloc=False, num_tests=10, verbose=True)

For more information, see `Installation <https://f0uriest.github.io/keras2c/installation.html>`_ and  `Usage <https://f0uriest.github.io/keras2c/usage.html>`_


Supported Layers
****************
- **Core Layers**: Dense, Activation, Dropout, Flatten, Input, Reshape, Permute, RepeatVector,  ActivityRegularization, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
- **Convolution Layers**: Conv1D, Conv2D, Conv3D, Cropping1D, Cropping2D, Cropping3D, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
- **Pooling Layers**: MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling3D,GlobalAveragePooling3D
- **Recurrent Layers**: SimpleRNN, GRU, LSTM, SimpleRNNCell, GRUCell, LSTMCell
- **Embedding Layers**: Embedding
- **Merge Layers**: Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
- **Advanced Activation Layers**: LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU
- **Normalization Layers**: BatchNormalization
- **Noise Layers**: GaussianNoise, GaussianDropout, AlphaDropout
- **Layer Wrappers**: TimeDistributed, Bidirectional
  
ToDo
****
- **Core Layers**: Lambda, Masking
- **Convolution Layers**: SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, Conv3DTranspose
- **Pooling Layers**: MaxPooling3D, AveragePooling3D
- **Locally Connected Layers**: LocallyConnected1D, LocallyConnected2D
- **Recurrent Layers**: ConvLSTM2D, ConvLSTM2DCell
- **Merge Layers**: Broadcasting merge between different sizes
- **Misc**: models made from submodels



Contribute
**********

- Documentation: `<https://f0uriest.github.io/keras2c/>`_
- Issue Tracker: `<https://github.com/f0uriest/keras2c/issues>`_
- Source Code: `<https://github.com/f0uriest/keras2c/>`_
  
License
*******

The project is licensed under the LGPLv3 license.


.. |Build-Status| image:: https://travis-ci.org/f0uriest/keras2c.svg?branch=master
    :target: https://travis-ci.org/f0uriest/keras2c
    :alt: Build Status
.. |Codecov| image:: https://codecov.io/gh/f0uriest/keras2c/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/f0uriest/keras2c
    :alt: Code Coverage
.. |License| image:: https://img.shields.io/github/license/f0uriest/keras2c
    :target: https://github.com/f0uriest/keras2c/blob/master/LICENSE
    :alt: License: LGPLv3
.. |DOI| image:: https://zenodo.org/badge/193152058.svg
    :target: https://zenodo.org/badge/latestdoi/193152058
    :alt: Please Cite Keras2c!

