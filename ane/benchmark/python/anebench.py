import os 
import shutil
import ctypes
import tempfile
import coremltools as ct


libANEBenchPath = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "capi", "libANEBench.dylib"))
libANEBench = ctypes.CDLL(libANEBenchPath)


def benchmark_coreml_model(model: ct.models.MLModel, iterations=100, mode="ane"):
    if not isinstance(model, ct.models.MLModel):
        raise ValueError("Provided model must be MLModel")
    
    model_path = tempfile.mkdtemp(suffix=".mlpackage")
    model.save(model_path)

    benchmark_native_func = libANEBench.benchmark_coreml_model
    benchmark_native_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    benchmark_native_func.restype = ctypes.c_int

    c_model_path = ctypes.create_string_buffer(str.encode(model_path))
    c_mode = ctypes.create_string_buffer(str.encode(mode))
    c_iters = ctypes.c_int(iterations)
    c_time_result = ctypes.c_double()
    res = benchmark_native_func(c_model_path, c_mode, c_iters, ctypes.byref(c_time_result))
    shutil.rmtree(model_path, ignore_errors=True)

    if res != 0:
        raise ValueError(f"Benchmark failed with status: {res.value}")

    return c_time_result.value


def benchmark_model(model, **kwargs):
    ct_model = ct.convert(model, convert_to="mlprogram", skip_model_load=True)
    return benchmark_coreml_model(ct_model, **kwargs)