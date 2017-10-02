OBJS = lapjv.o
CC = g++
DEBUG = -g
INCLUDES = -I../../../ # -I../../../../lap/
LIBS = -L/usr/local/lib/ -L../../../ad3/ # -L../../../../lap/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 -std=c++11 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : matching

matching : $(OBJS) matching.o
	$(CC) $(OBJS) matching.o $(LFLAGS) -o matching

matching.o: matching.cpp FactorMatching.h lapjv.h
	$(CC) $(CFLAGS) matching.cpp

lapjv.o: lapjv.cpp lapjv.h
	$(CC) $(CFLAGS) lapjv.cpp

clean:
	rm -f *.o *~ matching
