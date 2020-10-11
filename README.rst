#######
keras2c
#######

|Build-Status| |Codecov| |Codacy|

|License|


keras2c is a library for deploying keras neural networks in C99, using only standard libraries.
It is designed to be as simple as possible for real time applications.


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

- Issue Tracker: `<https://github.com/f0uriest/keras2c/issues>`_
- Source Code: `<https://github.com/f0uriest/keras2c/>`_
  
License
*******

The project is licensed under the MIT license.


.. |Build-Status| image:: https://travis-ci.org/f0uriest/keras2c.svg?branch=master
    :target: https://travis-ci.org/f0uriest/keras2c
    :alt: Build Status
.. |Codecov| image:: https://codecov.io/gh/f0uriest/keras2c/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/f0uriest/keras2c
    :alt: Code Coverage
.. |Codacy|  image:: https://api.codacy.com/project/badge/Grade/ac0b3f7d65a64a1f987463a81d2e1596
    :target: https://www.codacy.com/app/f0uriest/keras2c?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=f0uriest/keras2c&amp;utm_campaign=Badge_Grade  
    :alt: Code Quality
.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://github.com/f0uriest/keras2c/blob/master/LICENSE
    :alt: License: MIT

