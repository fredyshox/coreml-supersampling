CC=clang
CFLAGS:= -g -Wall -O0 -arch arm64
LDFLAGS:= -g -F /System/Library/PrivateFrameworks/ -framework Foundation -framework CoreML -framework IOSurface -framework QuartzCore -framework AppleNeuralEngine -arch arm64
OBJS:=Benchmark.o main.o ANEInferenceEngine.o ANEMultithreadInferenceEngine.o CoreMLInferenceEngine.o
TARGET:=ane_bench

$(TARGET):$(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o:%.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(TARGET) $(OBJS)
