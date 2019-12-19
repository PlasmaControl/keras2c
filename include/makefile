
CC=gcc
ifeq ($(origin CI),undefined)
	ccflags = -Ofast -std=c99 -I./include/
else
	ccflags = -g -Og -std=c99 --coverage -I./include/
endif

OBJ = \
	k2c_activations.o \
	k2c_convolution_layers.o \
	k2c_core_layers.o \
	k2c_embedding_layers.o \
	k2c_helper_functions.o \
	k2c_merge_layers.o \
	k2c_normalization_layers.o \
	k2c_pooling_layers.o \
	k2c_recurrent_layers.o
DEPS = k2c_include.h k2c_tensor_include.h

libkeras2c.a: $(OBJ)
	ar rcs libkeras2c.a $(OBJ)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(ccflags)

clean:
	rm *.o *.a
