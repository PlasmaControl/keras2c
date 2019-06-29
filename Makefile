
FLAGS = -gcov
CC = gcc $(FLAGS)

K2C = k2c_include.h \
	k2c_activations.h \
	k2c_convolution_layers.h \
	k2c_core_layers.h \
	k2c_helper_functions.h \
	k2c_merge_layers.h \
	k2c_pooling_layers.h \
	k2c_recurrent_layers.h \
FNAME = test1

all: test_suite clean



test_suite: model_test_suite.c
	$(CC) -o test_suite model1_test_suite.c -lm

clean:
	rm -f *.o
	echo Clean done

delete:
	rm ./test_suite
