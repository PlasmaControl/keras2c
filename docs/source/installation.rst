============
Installation
============


keras2c can be downloaded from github: https://github.com/f0uriest/keras2c

The Python requirements can be installed with pip:

.. code-block:: bash

    pip install -r requirements.txt

Alternatively, create a conda environment using the provided YAML file:

.. code-block:: bash

    conda env create -f environment.yml


Additional packages for building the documentation and running the tests are included in the conda environment, but can also be installed separately with pip:

.. code-block:: bash

    pip install -r docs/requirements.txt
    pip install -r tests/requirements.txt


By default, the tests compile code with ``gcc``. You can set the ``CC``
environment variable to use a different compiler, for example ``clang`` on macOS
or ``gcc`` from MSYS2/MinGW on Windows.  The makefile will fall back to ``gcc``
if ``CC`` is unset.  It is also recommended to install ``astyle`` to
automatically format the generated code.
