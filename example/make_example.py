# make_example.py, convert model.h5 to C files using keras2c library
import keras2c, tensorflow as tf
tf_major = int(tf.version.VERSION.split(".")[0])
if tf_major >= 2:
    tf.compat.v1.enable_eager_execution()
keras2c.keras2c_main.k2c("model.h5", "example")
