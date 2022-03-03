#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

void surfaceToImageWithReshape(IOSurfaceRef surface, size_t width, size_t height, size_t channels) {
    size_t bytesPerPixel = IOSurfaceGetBytesPerRow(surface);
    size_t bitsPerComponent = 16;
    size_t bitsPerPixel = bytesPerPixel * bitsPerComponent;
    size_t bytesPerRow = width * bytesPerPixel;

    void* dataPtr = IOSurfaceGetBaseAddress(surface);
    size_t dataSize = IOSurfaceGetAllocSize(surface);
    CFDataRef data = CFDataCreateWithBytesNoCopy(nil, dataPtr, dataSize, nil);
    CGDataProviderRef dataProvider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    CGImageRef image = CGImageCreate(
        width,
        height, 
        bitsPerComponent,
        bitsPerPixel, 
        bytesPerRow,
        colorSpace,
        kCGBitmapByteOrder16Little | kCGBitmapFloatComponents,
        dataProvider,
        nil,
        true,
        kCGRenderingIntentDefault
    );
    if (image == nil) {
        NSLog(@"Image nil!");
        return;
    }

    NSURL* url = [NSURL fileURLWithPath: @"./output.png"];
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL((CFURLRef) url, (CFStringRef)@"public.png", 1, nil);
    if (destination == nil) {
        NSLog(@"Destination nil!");
        return;
    }

    CGImageDestinationAddImage(destination, image, nil);
    BOOL res = CGImageDestinationFinalize(destination);
    NSLog(@"Image serialization status: %d", res);
    CGImageRelease(image);
}

void surfaceToImageWithReshapeSlow(IOSurfaceRef surface, size_t width, size_t height, size_t channels) {
    void* dataPtr = (void*) IOSurfaceGetBaseAddress(surface);
    size_t dataSize = IOSurfaceGetAllocSize(surface);
    size_t bytesPerPixel = IOSurfaceGetBytesPerRow(surface);

    size_t targetComponentCount = 4;
    size_t targetBitsPerComponent = 8;
    size_t targetBitsPerPixel = targetBitsPerComponent * targetComponentCount;
    size_t targetBytesPerRow = sizeof(uint8_t) * targetComponentCount * width;
    size_t newBufferSize = targetBytesPerRow * height;
    size_t newBufOffset = 0;
    uint8_t* newBuffer = (uint8_t*) malloc(newBufferSize);
    __fp16 compVal;
    for (size_t i = 0; i < dataSize; i += bytesPerPixel) {
        for (size_t c = 0; c < channels; c++) {
            compVal = *((__fp16*)(dataPtr + i) + c);
            *(newBuffer + newBufOffset + c) = (uint8_t)((float)compVal * 255);
        }
        newBufOffset += targetComponentCount;
    }
    
    CFDataRef data = CFDataCreateWithBytesNoCopy(nil, newBuffer, newBufferSize, nil);
    CGDataProviderRef dataProvider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    CGImageRef image = CGImageCreate(
        width,
        height, 
        targetBitsPerComponent,
        targetBitsPerPixel, 
        targetBytesPerRow,
        colorSpace,
        kCGImageAlphaNoneSkipLast,
        dataProvider,
        nil,
        true,
        kCGRenderingIntentDefault
    );
    if (image == nil) {
        NSLog(@"Image nil!");
        return;
    }

    NSURL* url = [NSURL fileURLWithPath: @"./output.png"];
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL((CFURLRef) url, (CFStringRef)@"public.png", 1, nil);
    if (destination == nil) {
        NSLog(@"Destination nil!");
        return;
    }

    CGImageDestinationAddImage(destination, image, nil);
    BOOL res = CGImageDestinationFinalize(destination);
    NSLog(@"Image serialization status: %d", res);
    CGImageRelease(image);
}