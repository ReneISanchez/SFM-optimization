CC=g++
LIB=libsfm.so
LIB_PATH=SfMLib
INC=$(LIB_PATH)/inc
FEAT=features

EIGEN_PATH=/usr/local/Cellar/eigen/3.2.2/include/eigen3


SOURCES=$(wildcard *.cpp)
OBJ=$(SOURCES:.cpp=.o)

CFLAGS=-I./$(INC) -I./$(INC)/$(FEAT) -I$(EIGEN_PATH) `pkg-config --cflags opencv` -Wall -O2 -g -pg
LDFLAGS=-L. -L./$(LIB_PATH) -lsfm `pkg-config --libs opencv`

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
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
		cd $(LIB_PATH) ; $(MAKE) clean ; cd .. ; rm *.o $(EXE)
