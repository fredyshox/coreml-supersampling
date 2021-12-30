CC=clang
CFLAGS:= -g -Wall -O0 -arch arm64
LDFLAGS:= -g -I. -F /System/Library/PrivateFrameworks/ -framework Foundation  -framework CoreML -framework AppleNeuralEngine -arch arm64
OBJS:=ANETestMLModel.o ANEClient+Intercept.o
TARGET:=ANETestMLModel

$(TARGET):$(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o:%.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(TARGET) $(OBJS)