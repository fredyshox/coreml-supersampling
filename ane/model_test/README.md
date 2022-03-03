# ANE ops compatibility test

A series of tests were performed using models with problematic ops, that might be incompatible. 

Tested ops are:
* Arithmetics (add/sub/mul) with same shape
* Arithmetics (add/sub/mul) with scalar (broadcasting 1)
* Arithmetics (add/sub/mul) with broadcastable shape (broadcasting 2)
* Cumulative sum
* Floor
* Gather
* Minimum & Maximum
* Split

## Setup

Model compatiblity is evaluated by swizzling `doEvaluateDirectWithModel:options:request:qos:error:` of `_ANEClient` to intercept `_ANEModel` input and output keys. If these match input/output keys of complete model, it means whole model is running on ANE (thus is compatible).

## Results

| Operation | Compatibility | Hardware |
| -------------------------- | :--------------------- | -- |
| Arithmetics (same-shape)   | :white_check_mark:     | M1 | 
| Arithmetics (scalar)       | :white_check_mark:     | M1 |
| Arithmetics (broadcasting) | :white_check_mark:     | M1 |
| Cumulative Sum             | :white_large_square: * | M1 |
| Floor                      | :white_large_square:   | M1 |
| Gather                     | :white_large_square:   | M1 |
| Minimum                    | :white_check_mark:     | M1 |
| Maximum                    | :white_check_mark:     | M1 |
| Split                      | :white_large_square:   | M1 |

\* model loads, but fails at execution