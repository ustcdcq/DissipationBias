# Compiler and linker settings
NVCC = nvcc
NVCCFLAGS = -arch=sm_60 -O3 -Xcompiler -Wall
LDFLAGS = -lcurand 

# Target executable name
TARGET = CC

# Object files to be created
OBJS = main.o CellList.o ForceComputing.o Integration.o  LocalDensity.o main.o  OperateTheProcess.o ParametersReset.o  PropertyCheck.o RandomNumber.o  Structure.o Clone.o

# Compile and link
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)
