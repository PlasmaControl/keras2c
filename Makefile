
FLAGS = -Ofast
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

all: predictor test_suite clean



test_suite: predictor.o test_suite.o 
	$(CC) $(FLAGS) -o test_suite test_suite.o predictor.o -lm

test_suite.o: $(FNAME)_test_suite.c $(K2C)
	$(CC) -c $(FNAME)_test_suite.c

test1.o: $(FNAME).c
	gcc $(FLAGS) -c test1.c

clean:
	rm -f *.o
	echo Clean done
