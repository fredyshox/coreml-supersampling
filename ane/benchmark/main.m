#import <Foundation/Foundation.h>
#import <string.h>
#import "Benchmark.h"
#import "InferenceEngine.h"
#import "ANEInferenceEngine.h"
#import "ANEMultithreadInferenceEngine.h"
#import "CoreMLInferenceEngine.h"

#define NUMBER_OF_ITERATIONS 1000

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s engine mlmodelc\n", argv[0]);
        printf("\nSupported engines: ane, mtane[N], coreml\n");
        return 1;
    }

    char* engineName = argv[1];
    NSURL* mlmodelURL = [NSURL fileURLWithPath: [NSString stringWithUTF8String: argv[2]]];

    Benchmark* benchmark = nil;
    id<InferenceEngine> engine = nil;
    NSTimeInterval meanTime;
    NSInteger fps;
    if (strcmp(engineName, "ane") == 0) {
        engine = [[ANEInferenceEngine alloc] initWithMLModelURL:mlmodelURL];
    } else if (strncmp(engineName, "mtane", 5) == 0) {
        int threadCount = atoi(engineName + 5);
        threadCount = threadCount > 0 ? threadCount : 2;
        engine = [[ANEMultithreadInferenceEngine alloc] initWithMLModelURL:mlmodelURL threadCount:threadCount];
    } else if (strcmp(engineName, "coreml") == 0) { 
        engine = [[CoreMLInferenceEngine alloc] initWithMLModelURL:mlmodelURL];
    } else {
        printf("Engine not supported %s\n", engineName);
        return 1;
    }

    benchmark = [[Benchmark alloc] initWithEngine:engine];
    meanTime = [benchmark runBenchmarkWithIterationCount:NUMBER_OF_ITERATIONS];
    fps = (NSInteger) (1.0 / meanTime);
    printf("%g\n", meanTime);
    printf("%ld fps\n", fps);

    return 0;
}