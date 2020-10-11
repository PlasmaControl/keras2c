============================================================================================
Keras2c: A simple library for converting Keras neural networks to real-time friendly C code.
============================================================================================

Abstract
********
With the growth of machine learning models and neural networks in measurement and control systems comes the need to deploy these models in a way that is compatible with existing systems. Existing options for deploying neural networks either introduce very high latency, requires expensive and time consuming work to integrate into existing code bases, or only support a very limited subset of model types. We have therefore developed a new method, called Keras2c, which is a simple library for converting Keras/TensorFlow neural network models into real time compatible C code. It supports a wide range of Keras layer and model types, including multidimensional convolutions, recurrent layers, well as multi-input/output models, and shared layers. Keras2c re-implements the core components of Keras/TensorFlow required for predictive forward passes through neural networks in pure C, relying only on standard library functions. The core functionality consists of only ~1200 lines of code, making it extremely lightweight and easy to integrate into existing codebases. Keras2c has been sucessfully tested in experiments and is currently in use on the plasma control system at the DIII-D National Fusion Facility at General Atomics in San Diego.

Motivation
**********
TensorFlow is one of the most popular libraries for developing and training neural networks, and contains a high level Python API called Keras that has become extremely popular due to its ease of use and rich feature set. As the use of machine learning and neural networks grows in the field of diagnostic and control systems, one of the central challenges remains how to deploy the resulting trained models in a way that can be easily integrated into existing systems, particularly for real time predictions using machine learning models. Given that most machine learning development traditionally takes place in Python, most deployment schemes involve calling out to a Python process (often running on a distant network connected server) and using the existing Python libraries to pass data through the model. This introduces large latency, and is generally not feasible for real time applications. Other options include rewriting the entire network using the existing TensorFlow C/C++ API, though this is extremely time consuming, and requires linking the resulting code against the full TensorFlow library, containing millions of lines of code and with a binary size up to several GB. The release of TensorFlow 2.0 contained a new possibility, called "TensorFlow Lite", a reduced library designed to run on mobile and IoT devices. However, TensorFlow Lite only supports a very limited subset of the full Keras API. Therefore, we present a new option, Keras2c, a simple library for converting Keras/TensorFlow neural network models into real time compatible C code.

Method
******

Keras2c consists of two primary components: a backend library of C functions that each implement a single layer of a neural net (eg, Dense, Conv2D, LSTM), and a Python script that generates C code to call the layer functions in the right order to implement the network. The total library of backend layer functions is only ~1200 lines of code, and uses only C standard library functions, yet covers a very wide range of Keras functionality, summarized below:

Supported Layers
################
- **Core Layers**: Dense, Activation, Flatten, Input, Reshape, Permute, RepeatVector
- **Convolution Layers**: Convolution (1D/2D/3D, with arbitrary stride/dilation/padding), Cropping (1D/2D/3D), UpSampling (1D/2D/3D), ZeroPadding (1D/2D/3D)
- **Pooling Layers**: MaxPooling (1D/2D/3D), AveragePooling (1D/2D/3D), GlobalMaxPooling (1D/2D/3D), GlobalAveragePooling (1D/2D/3D)
- **Recurrent Layers**: SimpleRNN, GRU, LSTM (statefull or stateless)
- **Embedding Layers**: Embedding
- **Merge Layers**: Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot
- **Normalization Layers**: BatchNormalization
- **Layer Wrappers**: TimeDistributed, Bidirectional
- **Activations**: ReLU, tanh, sigmoid, hard sigmoid, exponential, softplus, softmax, softsign, LeakyReLU, PReLU, ELU, ThresholdedReLU


.. figure:: flow_graph.png
    :align: center
    :scale: 50 %

    Workflow of converting Keras model to C code with Keras2C

The Keras2c Python script takes in a trained Keras model and extracts the weights and other parameters, and parses the graph structure to determine the order that functions should be called to obtain the correct results. It then generates  C code for a predictor function, that can be called with a set of inputs to generate predictions. It also generates helper functions for initializing and cleanup, to handle memory allocation (by default all variables are declared on the stack, though it also supports the option of dynamically allocating memory before execution). In addition to simple sequential models, Keras2c also supports more complicated architectures created using the Keras functional API, including multi-input/multi-output networks with complicated branching and merging internal structures.

To confirm that the generated code accurately reproduces the outputs of the original model, Keras2c also generates sample input/output pairs from the original network. It then automatically tests the generated code with the same inputs to verify that the generated code produces equivalent outputs.

Benchmarks
**********

Keras2c has also been benchmarked against Python Keras/TensorFlow for single CPU performance, and the generated code has been shown to be significantly faster for small to medium sized models.
(All tests conducted on Intel Core i7-8750H CPU @ 2.20GHz, single threaded, 32GB RAM. Keras2c compiled with GCC 7.4.0 with -O3 optimization. Python Keras v2.2.4, TensorFlowCPU v1.13.1, mkl v2019.1)

.. figure:: benchmarking.png
    :align: center

    Benchmarking results, Keras2c vs Keras/Tensorflow in Python. 

    
