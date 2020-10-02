PROJECT := build/sparse_matrix_transpose
SRC := $(wildcard src/*.cu)
OBJ := $(SRC:src/%.cu=build/%.o)
CXX := g++
NVCC := nvcc
CXXFLAGS = -std=c++11 -w
NVCCFAGS = -lcusparse
CFLAGS := -I include/ -c

all: $(PROJECT)

$(PROJECT): $(OBJ)
	$(NVCC) $(NVCCFAGS) $(OBJ) -o $(PROJECT)

build/%.o: src/%.cu
	mkdir -p build
	$(NVCC) $(CXXFLAGS) $(CFLAGS) $< -o $@

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