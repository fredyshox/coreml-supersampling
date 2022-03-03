#import <Foundation/Foundation.h>
#import "InferenceEngine.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEMultithreadInferenceEngine : NSObject <InferenceEngine>

@property (nonatomic, readonly) int threadCount;

- (id)initWithMLModelURL:(NSURL*)modelURL threadCount:(int)threadCount;
- (void)runInferenceOnDummyData;
- (int)passesPerIteration;

@end 

NS_ASSUME_NONNULL_END