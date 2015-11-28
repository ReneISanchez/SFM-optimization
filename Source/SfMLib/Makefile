CC=g++
SRC=src
INC=inc
FEAT=features

DEBUG_INFO=-g

SOURCES=$(wildcard $(SRC)/*.cpp $(SRC)/$(FEAT)/*.cpp)
OBJ=$(SOURCES:.cpp=.o)
#-march=native
CFLAGS=-fPIC -I./$(INC) -I./$(INC)/$(FEAT) `pkg-config --cflags opencv` -Wall -O3 -march=native -msse4.2 $(DEBUG_INFO)
LDFLAGS=-shared -L./$(SRC) `pkg-config --libs opencv`
EXE=libsfm.so

#CFLAGS+= -fopenmp
#LDFLAGS+= -lgomp

all: $(OBJ)
		$(CC) $(CFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS) ; mv $(EXE) ..

%.o: %.cpp
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
		rm $(SRC)/*.o $(SRC)/$(FEAT)/*.o ../$(EXE)
