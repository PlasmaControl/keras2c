CC_INC += ../include
CC_LIBS += ../include/libkeras2c.a
CC_FLAGS += -std=c99

all: example.o
	$(CC) example_test_suite.c example.o $(CC_LIBS) -lm -I$(CC_INC) $(CC_FLAGS) -o example

example.o:
	$(CC) example.c -c -lm -I$(CC_INC) $(CC_FLAGS)

clean:
	rm example *.o
