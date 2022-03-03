#ifndef ANEBench_CApi_h
#define ANEBench_CApi_h

/**
* Benchmark coreml model.
* @param mlmodelPath path to mlmodel/mlpackage
* @param mode inference mode, one of (ane, mtane, coreml)
* @param iters number of iterations 
* @param result pointer where time result will be stored (in seconds)
* @return status zero if successful
*/
int benchmark_coreml_model(char* mlmodelPath, char* mode, int iters, double* result);

/**
* Same as benchmark_coreml_model, but for already compiled models packaged in *.mlmodelc
*/
int benchmark_compiled_coreml_model(char* mlmodelcPath, char* mode, int iters, double* result);

#endif /* ANEBench_CApi_h */ 