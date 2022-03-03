CC=clang
CFLAGS:= -g -I.. -Wall -O0 -arch arm64
LDFLAGS:= -shared -g -F /System/Library/PrivateFrameworks/ -framework Foundation -framework CoreML -framework IOSurface -framework QuartzCore -framework AppleNeuralEngine -arch arm64
OBJS:= ../Benchmark.o ../ANEInferenceEngine.o ../ANEMultithreadInferenceEngine.o ../CoreMLInferenceEngine.o ANEBench_CApi.o
TARGET:=libANEBench.dylib

$(TARGET):$(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o:%.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(TARGET) $(OBJS)
