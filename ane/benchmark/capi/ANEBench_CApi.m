#include "ANEBench_CApi.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <string.h>

#import "Benchmark.h"
#import "InferenceEngine.h"
#import "ANEInferenceEngine.h"
#import "ANEMultithreadInferenceEngine.h"
#import "CoreMLInferenceEngine.h"

int _benchmark_compiled_coreml_model_at_url(NSURL* mlmodelcURL, char* mode, int iters, double* result) {
    Benchmark* benchmark = nil;
    id<InferenceEngine> engine = nil;
    double meanTime;
    if (strcmp(mode, "ane") == 0) {
        engine = [[ANEInferenceEngine alloc] initWithMLModelURL:mlmodelcURL];
    } else if (strncmp(mode, "mtane", 5) == 0) {
        int threadCount = atoi(mode + 5);
        threadCount = threadCount > 0 ? threadCount : 2;
        engine = [[ANEMultithreadInferenceEngine alloc] initWithMLModelURL:mlmodelcURL threadCount:threadCount];
    } else if (strcmp(mode, "coreml") == 0) { 
        engine = [[CoreMLInferenceEngine alloc] initWithMLModelURL:mlmodelcURL];
    } else {
        NSLog(@"Engine not supported %s\n", mode);
        return 1;
    }

    benchmark = [[Benchmark alloc] initWithEngine:engine];
    meanTime = [benchmark runBenchmarkWithIterationCount:iters];
    *result = meanTime;

    return 0;
}

int benchmark_coreml_model(char* mlmodelPath, char* mode, int iters, double* result) {
    NSError* error = nil;
    NSURL* mlmodelURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelPath]];
    NSURL* mlmodelcURL = [MLModel compileModelAtURL:mlmodelURL error:&error];
    if (error != nil) {
        NSLog(@"Model compilation error: %@", error);
        return 2;
    }

    int ret = _benchmark_compiled_coreml_model_at_url(mlmodelcURL, mode, iters, result);
    [[NSFileManager defaultManager] removeItemAtURL:mlmodelcURL error:nil];
    
    return ret;
}

int benchmark_compiled_coreml_model(char* mlmodelcPath, char* mode, int iters, double* result) {
    NSURL* mlmodelcURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlmodelcPath]];
    return _benchmark_compiled_coreml_model_at_url(mlmodelcURL, mode, iters, result);
}