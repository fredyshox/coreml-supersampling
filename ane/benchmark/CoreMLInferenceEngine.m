#import "CoreMLInferenceEngine.h"
#import <CoreML/CoreML.h>

#define ABORT_ON_ERR(err) if (err != nil) { NSLog(@"Error at %d: %@", __LINE__, err); abort(); }
#define ABORT_WITH_MESSAGE(...) NSLog(__VA_ARGS__); abort();

@implementation CoreMLInferenceEngine {
    MLModel* _model;
    id<MLFeatureProvider> _input;
}

- (id)initWithMLModelURL:(NSURL*)modelURL {
    self = [super init];
    if (self) {
        NSError* error = nil;

        MLModelConfiguration* configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsAll;

        MLModel* model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:&error];
        if (model == nil) {
            ABORT_WITH_MESSAGE(@"Failure when loading mlmodel: %@", error);
        }

        NSDictionary* inputDescriptionsByName = [[model modelDescription] inputDescriptionsByName];
        NSMutableDictionary* dictionaryInput = [NSMutableDictionary new];
        for (NSString* key in inputDescriptionsByName) {
            MLMultiArrayConstraint* constraint = [inputDescriptionsByName[key] multiArrayConstraint];
            if (constraint == nil) {
                ABORT_WITH_MESSAGE(@"Invalid mlmodel. Input not multi array.");
            }

            MLMultiArray* array = [[MLMultiArray alloc] initWithShape:constraint.shape dataType:constraint.dataType error:&error];
            ABORT_ON_ERR(error);

            dictionaryInput[key] = array;
        }
        MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dictionaryInput error:&error];
        ABORT_ON_ERR(error);

        _input = input;
        _model = model;
    }

    return self;
}

- (void)runInferenceOnDummyData {
    NSError* error = nil;
    [_model predictionFromFeatures: _input error:&error];
    ABORT_ON_ERR(error);
}

- (int)passesPerIteration {
    return 1;
}

@end