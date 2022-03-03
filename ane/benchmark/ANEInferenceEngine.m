#import "ANEInferenceEngine.h"
#import <IOSurface/IOSurface.h>
#import <CoreML/CoreML.h>

// AppleNeuralEngine.framework classes forward declarations
@class _ANEClient;
@class _ANEModel;
@class _ANERequest;
@class _ANEIOSurfaceObject;

#define ABORT_ON_ERR(res, err) if (!res) { NSLog(@"Error at %d: %@", __LINE__, err); abort(); }
#define ABORT_WITH_MESSAGE(...) NSLog(__VA_ARGS__); abort();

@implementation ANEInferenceEngine {
    MLModel* _model;
    _ANEClient* _client;
    _ANEModel* _aneModel; 
    _ANEIOSurfaceObject* _dummyInputData;
    _ANEIOSurfaceObject* _dummyOutputData;
}

- (id)initWithMLModelURL:(NSURL*)modelURL {
    self = [super init];
    if (self) {
        BOOL res; 
        NSError* error = nil;

        MLModel* model = [MLModel modelWithContentsOfURL:modelURL error:&error];
        if (model == nil) {
            ABORT_WITH_MESSAGE(@"Failure when loading mlmodel: %@", error);
        }

        if ([[[model modelDescription] inputDescriptionsByName] count] != 1) {
            ABORT_WITH_MESSAGE(@"Invalid input count. ANEInferenceEngine currently supports only one input");
        }
        if ([[[model modelDescription] outputDescriptionsByName] count] != 1) {
            ABORT_WITH_MESSAGE(@"Invalid output count. ANEInferenceEngine currently supports only one output");
        }

        NSString* inputName = [[[[model modelDescription] inputDescriptionsByName] allKeys] firstObject];
        NSString* outputName = [[[[model modelDescription] outputDescriptionsByName] allKeys] firstObject];

        MLFeatureDescription* inputDescription = [[model modelDescription] inputDescriptionsByName][inputName];
        MLFeatureDescription* outputDescription = [[model modelDescription] outputDescriptionsByName][outputName];
        MLMultiArrayConstraint* inputConstraint = [inputDescription multiArrayConstraint];
        MLMultiArrayConstraint* outputConstraint = [outputDescription multiArrayConstraint];
        if (inputConstraint == nil) {
            ABORT_WITH_MESSAGE(@"Invalid mlmodel. Input not multi array.");
        } else if (outputConstraint == nil) {
            ABORT_WITH_MESSAGE(@"Invalid mlmodel. Output not multi array.");
        }
        NSArray<NSNumber*>* inputShape = [inputConstraint shape];
        // output shape empty in mlmodels metadata...
        //NSArray<NSNumber*>* outputShape = [outputConstraint shape];
        NSNumber *ih = inputShape[1], *iw = inputShape[2], *ic = inputShape[3];
        //NSNumber *oh = outputShape[1], *ow = outputShape[2], *oc = outputShape[3];

        NSDictionary* modelKeyDict = @{
            @"isegment": @0,
            @"inputs": @{
                inputName: @{
                    @"shape": @[ic, ih, @1, iw, @1]
                }
            },
            @"outputs": @{
                outputName: @{
                    @"shape": @[@3, ih, @1, iw, @1]
                }
            }
        };
        NSData* modelKeyData = [NSJSONSerialization dataWithJSONObject:modelKeyDict options:0 error:nil];
        NSString* modelKey = [[NSString alloc] initWithData:modelKeyData encoding:NSUTF8StringEncoding];

        IOSurfaceRef inputSurface = IOSurfaceCreate((CFDictionaryRef) @{
            (NSString *) kIOSurfaceBytesPerElement: @2,
            (NSString *) kIOSurfaceHeight: @(ih.intValue * iw.intValue),
            (NSString *) kIOSurfaceWidth: ic,
            (NSString *) kIOSurfaceBytesPerRow: @(((ic.intValue / 64) + 1) * 64),
            (NSString *) kIOSurfacePixelFormat: @1278226536, // kCVPixelFormatType_OneComponent16Half
        });
        IOSurfaceRef outputSurface = IOSurfaceCreate((CFDictionaryRef) @{
            (NSString *) kIOSurfaceBytesPerElement: @2,
            (NSString *) kIOSurfaceHeight: @(ih.intValue * iw.intValue),
            (NSString *) kIOSurfaceWidth: @3,
            (NSString *) kIOSurfaceBytesPerRow: @(((3 / 64) + 1) * 64),
            (NSString *) kIOSurfacePixelFormat: @1278226536, // kCVPixelFormatType_OneComponent16Half
        });

        NSURL* milURL = [modelURL URLByAppendingPathComponent: @"model.mil"];
        _model = model;
        _client = [[_ANEClient alloc] initWithRestrictedAccessAllowed: NO];
        _aneModel = [_ANEModel modelAtURL:milURL key:modelKey];
        res = [_client compiledModelExistsFor: _aneModel];
        if (!res) {
            res = [_client compileModel: _aneModel options: @{} qos: QOS_CLASS_USER_INTERACTIVE error: &error];
            ABORT_ON_ERR(res, error);
        }

        res = [_client doLoadModel: _aneModel options: @{} qos: QOS_CLASS_USER_INTERACTIVE error: &error];
        ABORT_ON_ERR(res, error);

        _dummyInputData = [[_ANEIOSurfaceObject alloc] initWithIOSurface:inputSurface];
        _dummyOutputData = [[_ANEIOSurfaceObject alloc] initWithIOSurface:outputSurface];
    }

    return self;
}

- (void)runInferenceOnDummyData {
    NSError* error = nil;
    _ANERequest* request = [_ANERequest requestWithInputs:@[_dummyInputData] inputIndices:@[@0] outputs:@[_dummyOutputData] outputIndices:@[@0] perfStats:@[] procedureIndex:@0];
    BOOL res = [_client doEvaluateDirectWithModel:_aneModel options:@{} request:request qos:QOS_CLASS_USER_INTERACTIVE error:&error];
    if (!res) {
        NSLog(@"Warning: error while performing inference: %@", error);
    }
}

- (int)passesPerIteration {
    return 1;
}

@end