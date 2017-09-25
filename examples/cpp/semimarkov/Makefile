OBJS = 
CC = g++
DEBUG = -g
INCLUDES = -I../../../
LIBS = -L/usr/local/lib/ -L../../../ad3/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 -std=c++11 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : semimarkov

semimarkov : $(OBJS) semimarkov.o 
	$(CC) $(OBJS) semimarkov.o $(LFLAGS) -o semimarkov

semimarkov.o: semimarkov.cpp FactorBinarySegmentation.h
	$(CC) $(CFLAGS) semimarkov.cpp

clean:
	rm -f *.o *~ semimarkov
