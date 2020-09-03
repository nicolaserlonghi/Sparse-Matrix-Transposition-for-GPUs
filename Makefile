CXX = icpc
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -qopenmp
LDFLAGS  = -std=c++11 -qopenmp

TARGET =
ifdef ISA
ifeq ($(ISA),mic)
    TARGET = -mmic
else
ifeq ($(ISA),avx)
    TARGET = -xavx
else
ifeq ($(ISA),avx2)
    TARGET = -xCORE-AVX2
else
    TARGET = -xavx
endif # avx2
endif # avx
endif # mic
else
    TARGET = -xCORE-AVX2
endif # empty

CXXFLAGS += $(TARGET)
LDFLAGS  += $(TARGET)

all: sptrans.out

sptrans.out: main.cpp
	$(CXX) ${CXXFLAGS} $(LDFLAGS) $^ -o $@	-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

clean:
	rm -rf sptrans.out
