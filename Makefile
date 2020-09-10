PROJECT := build/sparse_matrix_transpose
SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:src/%.cpp=build/%.o)
LD := g++
CXX := g++
CXXFLAGS = -std=c++11 -w -O3
CFLAGS := -I include/ -c

all: $(PROJECT)

$(PROJECT): $(OBJ)
	$(LD) $(OBJ) -o $(PROJECT)

build/%.o: src/%.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(CFLAGS) $< -o $@

install:
	mkdir -p bin
	cp $(PROJECT) bin/

help:
	@echo all: compiles all files
	@echo install: installs application at right place
	@echo clean: delets everything except source file

clean:
	rm $(OBJ) $(PROJECT)
	
.PHONY: all clean install help