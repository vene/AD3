OBJS =
CC = g++
DEBUG = -g
INCLUDES = -I../../../
LIBS = -L/usr/local/lib/ -L../../../ad3/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 -fPIC $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : simple_grid test_distance

simple_grid : $(OBJS) simple_grid.o 
	$(CC) $(OBJS) simple_grid.o $(LFLAGS) -o simple_grid

simple_grid.o: simple_grid.cpp FactorSequence.h
	$(CC) $(CFLAGS) simple_grid.cpp

test_distance: $(OBJS) test_distance.o
	$(CC) $(OBJS) test_distance.o $(LFLAGS) -o test_distance

test_distance.o: test_distance.cpp FactorSequenceDistance.h
	$(CC) $(CFLAGS) test_distance.cpp

clean:
	rm -f *.o *~ simple_grid test_distance
