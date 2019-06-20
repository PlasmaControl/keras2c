FLAGS = -Ofast
all: test_suite clean



test_suite: test1.o test1_test_suite.o
	gcc $(FLAGS) -o test_suite test1_test_suite.o test1.o -lm

test1_test_suite.o: test1_test_suite.c
	gcc $(FLAGS) -c test1_test_suite.c

test1.o: test1.c
	gcc $(FLAGS) -c test1.c

clean:
	rm -f *.o
	echo Clean done
