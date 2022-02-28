#import <Foundation/Foundation.h>
#import "InferenceEngine.h"

NS_ASSUME_NONNULL_BEGIN

@interface CoreMLInferenceEngine : NSObject <InferenceEngine>

- (id)initWithMLModelURL:(NSURL*)modelURL;
- (void)runInferenceOnDummyData;
- (int)passesPerIteration;

@end 

NS_ASSUME_NONNULL_END