LIBS=-laf `pkg-config --libs opencv`
LIB_PATHS=-L/home/sfm-admin/arrayfire/build/lib
CFLAGS=`pkg-config --cflags opencv`
INCLUDES=-I/home/sfm-admin/arrayfire/build/include
CC=g++ $(COMPILER_OPTIONS)
COMPILER_OPTIONS=-std=c++11 -g

all: main.cpp Makefile
	$(CC) $(CFLAGS) main.cpp -o test $(INCLUDES) $(LIBS) $(LIB_PATHS)
