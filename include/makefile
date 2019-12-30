
CC=gcc

ifeq ($(CC), gcc)
	OPTFLAGS = -O3 -march=native -fwhole-program -fopt-info-vec -fopt-info-vec-missed
else ifeq ($(CC), icc)
	OPTFLAGS = -O3 -xHost -ipo -qopt-report=5 -qopt-report-phase=vec
else
	OPTFLAGS = -O3
endif

ifeq ($(origin CI),undefined)
	CCFLAGS = $(OPTFLAGS) -std=c99 -I./include/
else
	CCFLAGS = -g -Og -std=c99 --coverage -I./include/
endif

# -march=skylake-avx512

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

DEPS = \
	k2c_include.h \
	k2c_tensor_include.h

libkeras2c.a: $(OBJ)
	ar rcs libkeras2c.a $(OBJ)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

clean:
	rm *.o *.a
