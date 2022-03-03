#import <Foundation/Foundation.h>
#import "InferenceEngine.h"

NS_ASSUME_NONNULL_BEGIN

@interface Benchmark : NSObject 

- (id)initWithEngine:(id<InferenceEngine>)engine;
- (NSTimeInterval)runBenchmarkWithIterationCount:(int)iterations;

@end 

NS_ASSUME_NONNULL_END