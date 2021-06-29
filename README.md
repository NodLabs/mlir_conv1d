# MLIR Conv-1d Vectorization Experiments

This repo contains all the experiments used to evaluate options
for direct vectorization of 1d convolutions.


The code in this repo takes an mlir file, lowers it to LLVMIR,
extracts the assembly for the conv1d function and runs it through
llvm-mca.

## How to run

```
./run.py -m [path to mlir build dir] -o [option name]
```

Currently, the supported options are
- scalar
- scalar_unrolled
- multi_reduction
- unrolled_contraction
- shuffled_contraction
