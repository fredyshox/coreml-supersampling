#import <Foundation/Foundation.h>
#import <objc/runtime.h>

@class _ANEClient;
@class _ANEModel;
@class _ANERequest;

typedef void (^_ANEInterceptorCallback)(BOOL);

@interface NSObject (ANEClientInterceptor)
+ (void)swizzleInterceptorWithInputName: (NSString*) inputName outputName: (NSString*) outputName callback: (_ANEInterceptorCallback) callback; 
- (BOOL)doEvaluateModelWithInterceptor: (_ANEModel *) model options: (NSDictionary *) options request: (_ANERequest *) request qos: (dispatch_qos_class_t) qos error: (NSError**) errorPtr;
@end