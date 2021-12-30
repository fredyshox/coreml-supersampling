#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <stdio.h>

#import "ANEClient+Intercept.h"

@class _ANEClient;

#define ANE_MODEL_SUCCESS 0x0
#define ANE_MODEL_FAILURE 0xf

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s modelpath\n", argv[0]);
        return 1;
    }

    NSURL* modelUrl = [NSURL fileURLWithPath: [NSString stringWithUTF8String: argv[1]]];
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    [config setComputeUnits: MLComputeUnitsAll];
    NSError* error = nil;

    MLModel* model = [MLModel modelWithContentsOfURL:modelUrl configuration: config error:&error];
    if (error != nil) {
        NSLog(@"Failed to load mlmodel: %@", [error localizedDescription]);
        return 2;
    }

    NSString* inputName = [[[[model modelDescription] inputDescriptionsByName] allKeys] firstObject];
    NSString* outputName = [[[[model modelDescription] outputDescriptionsByName] allKeys] firstObject];

    MLFeatureDescription* inputDescription = [[model modelDescription] inputDescriptionsByName][inputName];
    MLMultiArrayConstraint* inputConstraint = [inputDescription multiArrayConstraint];
    if (inputConstraint == nil) {
        NSLog(@"Something wrong with provided model. Input not multi array.");
        return 3;
    }

    NSArray<NSNumber*>* inputShape = [inputConstraint shape];
    MLMultiArrayDataType inputType = [inputConstraint dataType];

    MLMultiArray* input = [[MLMultiArray alloc] initWithShape: inputShape
                                                     dataType: inputType
                                                        error: &error];
    if (error != nil) {
        NSLog(@"Error while initializing multiarray: %@", error.localizedDescription);
        return 4;
    }
    MLDictionaryFeatureProvider* inputProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary: @{inputName: input}
                                                                                                   error: &error];
    if (error != nil) {
        NSLog(@"Error while initializing dictionary provider: %@", error.localizedDescription);
        return 5;
    }

    dispatch_once_t reportResultOnce;
    __block int interceptorReturnValue = ANE_MODEL_FAILURE;
    [_ANEClient swizzleInterceptorWithInputName: inputName outputName: outputName callback: ^(BOOL result){
        dispatch_once(&reportResultOnce, ^{
            interceptorReturnValue = (result) ? ANE_MODEL_SUCCESS : ANE_MODEL_FAILURE;
            NSLog(@"ANE compatibility status: %d", result);
        });
    }];

    id<MLFeatureProvider> output = [model predictionFromFeatures: inputProvider 
                                                         options: [[MLPredictionOptions alloc] init] 
                                                           error: &error];
    if (error != nil) {
        NSLog(@"Error while running prediction: %@", error.localizedDescription);
        return 6;
    }

    NSLog(@"Done with status: %d!", interceptorReturnValue);

    return interceptorReturnValue;
}