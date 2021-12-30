#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <IOSurface/IOSurface.h>
#import <QuartzCore/QuartzCore.h>
#import <stdint.h>

#import "SurfaceToPng.h"

// AppleNeuralEngine.framework classes forward declarations
@class _ANEClient;
@class _ANEModel;
@class _ANERequest;
@class _ANEIOSurfaceObject;

#define ABORT_ON_ERR(res, err) if (!res) { NSLog(@"Error at %d: %@", __LINE__, err); abort(); }

NSString* IOSurfaceDescription(IOSurfaceRef surf) {
    uint32_t identifier = IOSurfaceGetID(surf);
    size_t planeCount = IOSurfaceGetPlaneCount(surf);
    size_t height = IOSurfaceGetHeight(surf);
    size_t width = IOSurfaceGetWidth(surf);
    int32_t subsampling = IOSurfaceGetSubsampling(surf);
    size_t bytesPerElement = IOSurfaceGetBytesPerElement(surf);
    size_t bytesPerRow = IOSurfaceGetBytesPerRow(surf);
    size_t elementHeight = IOSurfaceGetElementHeight(surf);
    size_t elementWidth = IOSurfaceGetElementWidth(surf);
    size_t allocSize = IOSurfaceGetAllocSize(surf);
    void* baseAddress = IOSurfaceGetBaseAddress(surf);
    size_t componentCount;

    NSMutableString* desc = [NSMutableString string];
    [desc appendFormat: @"IOSurface<%p> {\n", surf];
    [desc appendFormat: @"\tid: %u\n", identifier];
    [desc appendFormat: @"\tplaneCount: %lu\n", planeCount];
    [desc appendFormat: @"\theight: %lu\n", height];
    [desc appendFormat: @"\twidth: %lu\n", width];
    [desc appendFormat: @"\tsubsampling: %d\n", subsampling];
    [desc appendFormat: @"\tbytesPerElement: %lu\n", bytesPerElement];
    [desc appendFormat: @"\tbytesPerRow: %lu\n", bytesPerRow];
    [desc appendFormat: @"\telementHeight: %lu\n", elementHeight];
    [desc appendFormat: @"\telementWidth: %lu\n", elementWidth];
    [desc appendFormat: @"\tallocSize: %lu\n", allocSize];
    [desc appendFormat: @"\tbufAddress: %p\n", baseAddress];
    for (size_t i = 0; i < planeCount; i++) {
        height = IOSurfaceGetHeightOfPlane(surf, i);
        width = IOSurfaceGetWidthOfPlane(surf, i);
        elementHeight = IOSurfaceGetElementHeightOfPlane(surf, i);
        elementWidth = IOSurfaceGetElementWidthOfPlane(surf, i);
        componentCount = IOSurfaceGetNumberOfComponentsOfPlane(surf, i);
        bytesPerElement = IOSurfaceGetBytesPerElementOfPlane(surf, i);
        bytesPerRow = IOSurfaceGetBytesPerRowOfPlane(surf, i);
        baseAddress = IOSurfaceGetBaseAddressOfPlane(surf, i);

        [desc appendFormat: @"\tplane_%lu:\n", i];
        [desc appendFormat: @"\t\theight: %lu\n", height];
        [desc appendFormat: @"\t\twidth: %lu\n", width];
        [desc appendFormat: @"\t\tbytesPerElement: %lu\n", bytesPerElement];
        [desc appendFormat: @"\t\tbytesPerRow: %lu\n", bytesPerRow];
        [desc appendFormat: @"\t\telementHeight: %lu\n", elementHeight];
        [desc appendFormat: @"\t\telementWidth: %lu\n", elementWidth];
        [desc appendFormat: @"\t\tcomponentCount: %lu\n", componentCount];
        [desc appendFormat: @"\t\tbufAddress: %p\n", baseAddress];
    }
    [desc appendString: @"}\n"];

    return desc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        NSLog(@"Usage: program mlmodelc");
        return 1;
    }

    NSString* modelPath = [NSString stringWithUTF8String: argv[1]]; 
    NSURL* modelUrl = [NSURL fileURLWithPath: modelPath];
    NSURL* milUrl = [modelUrl URLByAppendingPathComponent: @"model.mil"];
    NSError* error = nil;
    BOOL res = false; 

    // initialization
    [_ANEClient initialize];
    [_ANEClient sharedConnection];

    // setup client
    _ANEClient* client = [[_ANEClient alloc] initWithRestrictedAccessAllowed: NO];

    // create model object
    NSString* modelKey = @"{\"isegment\":0,\"inputs\":{\"input_1\":{\"shape\":[36,1280,1,720,1]}},\"outputs\":{\"Identity\":{\"shape\":[3,1280,1,720,1]}}}";
    _ANEModel* model = [_ANEModel modelAtURL: milUrl key: modelKey];

    NSLog(@"Compiling model");
    res = [client compiledModelExistsFor: model];
    NSLog(@"Compiled model exists: %d key: %@", res, [model key]);
    if (!res) {
        res = [client compileModel: model options: @{} qos: QOS_CLASS_USER_INTERACTIVE error: &error];
        ABORT_ON_ERR(res, error);
    }

    // load model
    NSLog(@"Loading model");
    NSLog(@"Model before load: %@", [model description]);
    res = [client doLoadModel: model options: @{} qos: QOS_CLASS_USER_INTERACTIVE error: &error];
    ABORT_ON_ERR(res, error);
    NSLog(@"Model after load: %@", [model description]);

    // evaluation 
    NSLog(@"Running evaluate");

    // IO Surfaces - memory to share with ane dameon via xpc
    IOSurfaceRef inSurf = IOSurfaceCreate((CFDictionaryRef) @{
        (NSString *) kIOSurfaceBytesPerElement: @2,
        (NSString *) kIOSurfaceBytesPerRow: @128,
        (NSString *) kIOSurfaceHeight: @921600,
        (NSString *) kIOSurfacePixelFormat: @1278226536, // kCVPixelFormatType_OneComponent16Half
        (NSString *) kIOSurfaceWidth: @36
    });
    size_t inSurfAllocSize = IOSurfaceGetAllocSize(inSurf);
    NSLog(@"Created input iosurface\n%@", IOSurfaceDescription(inSurf));

    IOSurfaceLock(inSurf, 0, nil);
    uint8_t* inSurfAddr = (uint8_t*) IOSurfaceGetBaseAddress(inSurf); 
    for (int i = 0; i < inSurfAllocSize; i += 2) {
        // 0x3f00 --> float16 of 0.5
        inSurfAddr[i] = 0x00; 
        inSurfAddr[i+1] = 0x38;
    }
    IOSurfaceUnlock(inSurf, 0, nil);

    IOSurfaceRef outSurf = IOSurfaceCreate((CFDictionaryRef) @{
        (NSString *) kIOSurfaceBytesPerElement: @2,
        (NSString *) kIOSurfaceBytesPerRow: @64,
        (NSString *) kIOSurfaceHeight: @921600,
        (NSString *) kIOSurfacePixelFormat: @1278226536, // kCVPixelFormatType_OneComponent16Half
        (NSString *) kIOSurfaceWidth: @3
    });
    uint8_t* outSurfAddr = (uint8_t*) IOSurfaceGetBaseAddress(outSurf); 
    size_t outSurfAllocSize = IOSurfaceGetAllocSize(outSurf);
    NSLog(@"Created output iosurface\n%@", IOSurfaceDescription(outSurf));

    // create ane request
    _ANEIOSurfaceObject* aneInputSurface = [[_ANEIOSurfaceObject alloc] initWithIOSurface: inSurf];
    _ANEIOSurfaceObject* aneOutputSurface = [[_ANEIOSurfaceObject alloc] initWithIOSurface: outSurf];
    _ANERequest* request = [_ANERequest requestWithInputs: @[aneInputSurface] inputIndices: @[@0] outputs: @[aneOutputSurface] outputIndices: @[@0] perfStats: @[] procedureIndex: @0];
    [request setCompletionHandler: nil];
    NSLog(@"Created ANERequest: %@", request);

    // cold start
    NSLog(@"Cold start single run");
    res = [client doEvaluateDirectWithModel: model options: @{} request: request qos: QOS_CLASS_USER_INTERACTIVE error: &error];
    ABORT_ON_ERR(res, error);

    // benchmark
    NSLog(@"Starting benchmark");
    NSMutableArray<NSNumber*>* durationStorage = [[NSMutableArray alloc] initWithCapacity:1000];
    CFTimeInterval startTime, endTime, duration;
    uint32_t prevOutSurfSeed = IOSurfaceGetSeed(outSurf);
    uint32_t outSurfSeed = prevOutSurfSeed;
    for (int i = 0; i < 10; i++) {
        // iosurf changes, to prevent possible optimizations based on seed value
        // IOSurfaceLock(inSurf, 0, nil);
        // *inSurfAddr = 0x01;
        // *(inSurfAddr + 1) = (uint8_t) i;
        // IOSurfaceUnlock(inSurf, 0, nil);

        IOSurfaceLock(inSurf, 0, nil);
        for (int i = 0; i < inSurfAllocSize/2; i++) {
            *(inSurfAddr + i) = (__fp16)0.000001*i;
        }
        NSLog(@"Done cpu store..");
        IOSurfaceUnlock(inSurf, 0, nil);

        // evaluation with timing
        startTime = CACurrentMediaTime();

        BOOL res = [client doEvaluateDirectWithModel: model options: @{} request: request qos: QOS_CLASS_USER_INTERACTIVE error: &error];
        ABORT_ON_ERR(res, error);  

        endTime = CACurrentMediaTime();
        duration = endTime - startTime;
        [durationStorage addObject:[NSNumber numberWithDouble:duration]];

        // check outsurf seed if changed
        outSurfSeed = IOSurfaceGetSeed(outSurf);
        NSLog(@"Old seed: %d New seed: %d Diff %d", prevOutSurfSeed, outSurfSeed, (int) prevOutSurfSeed != outSurfSeed);
        prevOutSurfSeed = outSurfSeed;

        // IOSurfaceLock(outSurf, 0, nil);
        // __fp16 qqqq = *(__fp16*)(outSurfAddr + 4);
        // NSLog(@"Val %g", (float) qqqq);
        // for (int i = 0; i < 64; i++) {
        //     NSLog(@"-- %x", *(outSurfAddr + i));
        // }
        // IOSurfaceUnlock(outSurf, 0, nil);
    }

    NSTimeInterval sum = 0;
    NSTimeInterval min = MAXFLOAT;
    NSTimeInterval max = 0;
    NSTimeInterval current;
    for (NSNumber* n in durationStorage) {
        current = [n doubleValue];
        sum += current;
        if (current < min) {
            min = current;
        }
        if (current > max) {
            max = current;
        }
    }
    NSTimeInterval mean = sum / durationStorage.count;
    NSLog(@"Execution time stats: Min = %g, Max = %g, Mean = %g", min, max, mean);
    NSLog(@"Performance test result: %g (~%g fps)", mean, round(1.0 / mean));
    NSLog(@"All times %@", durationStorage);

    surfaceToImageWithReshapeSlow(outSurf, 1280, 720, 3);

    NSLog(@"Done");
    return 0; 
}
