#import "ANEClient+Intercept.h"

static char InterceptorKey;

@interface _ANEInterceptor: NSObject 
@property (readonly, nonatomic, strong) NSString* inputName;
@property (readonly, nonatomic, strong) NSString* outputName; 
- (instancetype)initWithInput: (NSString*) input output: (NSString*) output callback: (_ANEInterceptorCallback) callback;
- (void)notifyWithResult: (BOOL) result;
@end

@implementation _ANEInterceptor {
    _ANEInterceptorCallback _callback;
}

- (instancetype)initWithInput: (NSString*) input output: (NSString*) output callback: (_ANEInterceptorCallback) callback {
    self = [super init];
    if (self) {
        _inputName = input;
        _outputName = output;
        _callback = callback;
    }
    
    return self;
}

- (void)notifyWithResult: (BOOL) result {
    _callback(result);
}
@end

@implementation NSObject (ANEClientInterceptor)
- (BOOL)doEvaluateModelWithInterceptor: (_ANEModel *) model options: (NSDictionary *) options request: (_ANERequest *) request qos: (dispatch_qos_class_t) qos error: (NSError**) errorPtr {
    NSLog(@"Intercepted loading model with key: %@", [model key]);
    _ANEInterceptor* interceptor = objc_getAssociatedObject([self class], &InterceptorKey);
    NSError* error = nil;

    NSData* keyData = [[model key] dataUsingEncoding: NSUTF8StringEncoding];
    NSDictionary* parsedKey = [NSJSONSerialization JSONObjectWithData: keyData options: 0 error: &error];
    if (error != nil || parsedKey == nil) {
        NSLog(@"Unable to parse model key!");
    } else {
        NSArray<NSString*>* inputKeys = [parsedKey[@"inputs"] allKeys];
        NSArray<NSString*>* outputKeys = [parsedKey[@"outputs"] allKeys];
        BOOL inputValid = inputKeys.count == 1 && [inputKeys containsObject: interceptor.inputName];
        BOOL outputValid = outputKeys.count == 1 && [outputKeys containsObject: interceptor.outputName];
        [interceptor notifyWithResult: inputValid && outputValid];
    }

    // call yourself, but as implementations are exchanged this is original implementation
    return [self doEvaluateModelWithInterceptor: model options: options request: request qos: qos error: errorPtr];
}

+ (void)swizzleInterceptorWithInputName: (NSString*) inputName outputName: (NSString*) outputName callback: (_ANEInterceptorCallback) callback {
    Class class = [self class];

    SEL originalSelector = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
    SEL swizzledSelector = @selector(doEvaluateModelWithInterceptor:options:request:qos:error:);

    Method originalMethod = class_getInstanceMethod(class, originalSelector);
    Method swizzledMethod = class_getInstanceMethod(class, swizzledSelector);

    method_exchangeImplementations(originalMethod, swizzledMethod);

    _ANEInterceptor* interceptor = [[_ANEInterceptor alloc] initWithInput: inputName 
                                                                   output: outputName 
                                                                 callback: callback];
    objc_setAssociatedObject(self, &InterceptorKey, interceptor, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}
@end