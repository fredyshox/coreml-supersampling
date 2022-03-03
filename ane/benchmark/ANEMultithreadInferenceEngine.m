#import "ANEMultithreadInferenceEngine.h"
#import "ANEInferenceEngine.h"

#define TASK_NOT_DONE 0
#define TASK_DONE     1

/**
* Based on reply on issue 1356 "Inference of mlmodel not fully utilize CPU&ANE" on coremltools
* 
* "So, basically in single "thread" inference not used all available resources. 
*  But if I run inference in 2 or 3 threads (separate processes) 
*  I got 2x-2.5x speed up throughput with higher utilization aall available resources of CPU and ANE."
* 
* Source: https://github.com/apple/coremltools/issues/1356#issuecomment-979229166
*/
@implementation ANEMultithreadInferenceEngine {
    NSArray<ANEInferenceEngine*>* _childEngines;
}

- (id)initWithMLModelURL:(NSURL*)modelURL threadCount:(int)threadCount {
    self = [super init];
    if (self) {
        _threadCount = threadCount;
        NSMutableArray<ANEInferenceEngine*>* childEngines = [NSMutableArray new];
        for (int i = 0; i < threadCount; i++) {
            ANEInferenceEngine* engine = [[ANEInferenceEngine alloc] initWithMLModelURL:modelURL];
            [childEngines addObject:engine];
        }
        _childEngines = childEngines;
    }

    return self;
}

- (void)runInferenceOnDummyData {
    NSMutableArray<NSThread*>* threads = [NSMutableArray new];
    NSMutableArray<NSConditionLock*>* locks = [NSMutableArray new];
    for (int i = 0; i < _threadCount; i++) {
        ANEInferenceEngine* engine = _childEngines[i];
        NSConditionLock* lock = [[NSConditionLock alloc] initWithCondition:TASK_NOT_DONE];
        NSThread* thread = [[NSThread alloc] initWithBlock: ^void(){
            [lock lock];
            [engine runInferenceOnDummyData];
            [lock unlockWithCondition:TASK_DONE];
        }];

        [thread start];
        [threads addObject:thread];
        [locks addObject:lock];
    }

    for (NSConditionLock* lock in locks) {
        [lock lockWhenCondition:TASK_DONE];
        [lock unlock];
    }
}

- (int)passesPerIteration {
    return _threadCount;
}

@end