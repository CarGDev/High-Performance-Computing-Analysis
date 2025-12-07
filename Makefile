# Makefile for HPC Matrix Alignment Prototype
# Demonstrates memory alignment optimization impact

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -ffast-math -mavx -mfma
LDFLAGS = -lm
TARGET = matrix_alignment_prototype
SOURCE = matrix_alignment_prototype.c

# Default target
all: $(TARGET)

# Build the prototype
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Run the prototype
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

.PHONY: all run clean debug

