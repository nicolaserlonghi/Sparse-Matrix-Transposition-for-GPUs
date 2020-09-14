PROJECT := build/sparse_matrix_transpose
SRC := $(wildcard src/*.cu)
OBJ := $(SRC:src/%.cu=build/%.o)
CXX := g++
NVCC := nvcc
# Qui non serve -lcusparse
CXXFLAGS = -std=c++11 -w -O3 -arch=sm_62
NVCCFAGS = -lcusparse
CFLAGS := -I include/ -c

all: $(PROJECT)

# Qui serve -lcusparse altrimenti non funziona. PERCHE'??
$(PROJECT): $(OBJ)
	$(NVCC) $(NVCCFAGS) $(OBJ) -o $(PROJECT)

# Qui non serve -lcusparse
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