#import "Benchmark.h"
#import <CoreFoundation/CoreFoundation.h>
#import <QuartzCore/QuartzCore.h>

@implementation Benchmark {
    id<InferenceEngine> _inferenceEngine;
}

- (id)initWithEngine:(id<InferenceEngine>)engine {
    self = [super init];
    if (self) {
        _inferenceEngine = engine;
    }

    return self;
}

- (NSTimeInterval)runBenchmarkWithIterationCount:(int)iterations {
    NSMutableArray<NSNumber*>* durationStorage = [[NSMutableArray alloc] initWithCapacity:iterations];
    CFTimeInterval startTime, endTime, duration;
    for (int i = 0; i < iterations; i += [_inferenceEngine passesPerIteration]) {
        startTime = CACurrentMediaTime();

        [_inferenceEngine runInferenceOnDummyData];

        endTime = CACurrentMediaTime();
        duration = (endTime - startTime);
        [durationStorage addObject:[NSNumber numberWithDouble:duration]];
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
    NSTimeInterval mean = sum / (durationStorage.count * [_inferenceEngine passesPerIteration]);

    return mean;
}

@end 