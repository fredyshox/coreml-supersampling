CC=clang
CFLAGS:= -g -Wall -O0 -arch arm64
LDFLAGS:= -g -I. -F /System/Library/PrivateFrameworks/ -framework Foundation -framework AppleNeuralEngine -framework IOSurface -framework QuartzCore -framework CoreGraphics -framework ImageIO -arch arm64
OBJS:=ane_xpc.o
TARGET:=ane_xpc

$(TARGET):$(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o:%.m
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(TARGET) $(OBJS)
