CC=clang
CFLAGS:= -g -Wall -O0 -arch arm64
LDFLAGS:= -g -I. -F /System/Library/PrivateFrameworks/ -framework Foundation  -framework CoreML -framework IOSurface -framework AppleNeuralEngine -arch arm64
OBJS:=main.o ANEClient+Intercept.o
TARGET:=ane_test

$(TARGET):$(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o:%.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(TARGET) $(OBJS)