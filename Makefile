# Guide:
# make       # compile all binaries
# make clean # remove ALL binaries

.PHONY: all clean

# Default Compilers
CC  ?= gcc
CXX ?= g++

# Choose which OpenCV pkg-config name to use: opencv4 (modern) or opencv (older)
OPENCV_PKG ?= opencv4

# Language standard
CXXSTD ?= -std=c++17

# Linker & compiler flags
LDFLAGS  ?=
CFLAGS   ?= -O0 -Wall -Wextra
CXXFLAGS ?= -O0 -Wall -Wextra

# Add OpenCV flags (no-op if pkg-config can't find it)
CXXFLAGS += $(CXXSTD) $(shell pkg-config --cflags $(OPENCV_PKG))
LDFLAGS  += $(shell pkg-config --libs   $(OPENCV_PKG))

# Automated

# Grab .c and .cpp files
SOURCE_C   := $(wildcard *.c)
SOURCE_CXX := $(wildcard *.cpp)

# Map sources to binaries (remove extensions)
BIN_C   := $(SOURCE_C:.c=)
BIN_CXX := $(SOURCE_CXX:.cpp=)
BINS    := $(BIN_C) $(BIN_CXX)

all: $(BINS)

# IMPORTANT: OpenCV is C++ only. We compile/link everything with CXX so
# code that includes <opencv2/...> works even if the file extension is .c.
%: %.c
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	@echo "Cleaning"
	@rm -f $(BINS)