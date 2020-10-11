========================
Python API Documentation
========================

Main
****
.. autofunction:: keras2c.keras2c_main.k2c
.. autofunction:: keras2c.keras2c_main.model2c
.. autofunction:: keras2c.keras2c_main.gen_function_reset
.. autofunction:: keras2c.keras2c_main.gen_function_initialize
.. autofunction:: keras2c.keras2c_main.gen_function_terminate


Writing Layers
**************

.. autoclass:: keras2c.layer2c.Layers2C
    :members:
    :undoc-members:

Writing Weights
***************

.. autoclass:: keras2c.weights2c.Weights2C
    :members:
    :undoc-members:

Checking Model
**************
.. autofunction:: keras2c.check_model.is_valid_c_name
.. autofunction:: keras2c.check_model.name_check
.. autofunction:: keras2c.check_model.layers_supported_check
.. autofunction:: keras2c.check_model.activation_supported_check
.. autofunction:: keras2c.check_model.config_supported_check
.. autofunction:: keras2c.check_model.check_model

Graph Parsing
*************
.. autofunction:: keras2c.io_parsing.layer_type
.. autofunction:: keras2c.io_parsing.get_all_io_names
.. autofunction:: keras2c.io_parsing.get_layer_num_io
.. autofunction:: keras2c.io_parsing.get_layer_io_names
.. autofunction:: keras2c.io_parsing.get_model_io_names
.. autofunction:: keras2c.io_parsing.flatten

Test Generation
***************
.. autofunction:: keras2c.make_test_suite.make_test_suite

