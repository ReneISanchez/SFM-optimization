CC=g++
LIB=libsfm.so
LIB_PATH=SfMLib
INC=$(LIB_PATH)/inc
FEAT=features

EIGEN_PATH=/usr/include/eigen3


SOURCES=$(wildcard *.cpp)
OBJ=$(SOURCES:.cpp=.o)

#CCFLAGS+= -msse4.2 # enable sse4.2 intrinsic functions
CFLAGS=-I/home/sfm-admin/arrayfire/build/include -I./$(INC) -I./$(INC)/$(FEAT) -D__SSE4_2__ -I$(EIGEN_PATH) `pkg-config --cflags opencv` -Wall -O2 -g -pg -msse4.2
LDFLAGS= -laf -L/home/sfm-admin/arrayfire/build/lib -lcuda -L. -L./$(LIB_PATH) -lsfm `pkg-config --libs opencv`

ifeq ($(shell uname -s),Linux)
LDFLAGS+= -Wl,-rpath=./
endif

EXE=sfm

#CFLAGS+= -fopenmp
#LDFLAGS+= -lgomp

all: $(LIB) $(OBJ) 
		$(CC) $(CFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS)

$(LIB):
		cd $(LIB_PATH) ; $(MAKE) ; cd ..

%.o: %.cpp
		$(CC) $(CFLAGS) -o $@ -c -lpthread $<

clean:
		cd $(LIB_PATH) ; $(MAKE) clean ; cd .. ; rm *.o $(EXE)
