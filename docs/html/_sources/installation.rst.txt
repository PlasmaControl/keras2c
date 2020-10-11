============
Installation
============


keras2c can be downloaded from github: https://github.com/f0uriest/keras2c

The Python requirements can be installed with pip:

.. code-block:: bash

    pip install -r requirements.txt


Additional packages are required for building the documentation and running the tests, which can also be installed with pip:

.. code-block:: bash

    pip install -r docs/requirements.txt
    pip install -r tests/requirements.txt


By default, the tests compile code with ``gcc``. This can be modified to use a different compiler by changing the variable ``CC`` in ``tests/test_core_layers.py`` and ``include/makefile``
It is also recommended to install ``astyle`` to automatically format the generated code.
