#import <Foundation/Foundation.h>

@protocol InferenceEngine <NSObject>

- (void)runInferenceOnDummyData;
- (int)passesPerIteration;

@end
