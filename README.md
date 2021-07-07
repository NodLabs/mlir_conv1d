# MLIR Conv-1d Vectorization Experiments

This repo contains all the experiments used to evaluate options
for direct vectorization of 1d convolutions.


The code in this repo takes an mlir file, lowers it to LLVMIR,
extracts the assembly for the conv1d function and runs it through
llvm-mca.

## How to run

```
./run.py -m [path to mlir build dir] -o [option name] [-v vector_width] [-a arch] [-c cpu]
```

Examples:
```
./run.py -m ${LLVM_BUILD_DIR} -o multi_reduction -v 128
./run.py -m ${LLVM_BUILD_DIR} -o scalar -a arm64 -c cortex-a53
```

To see the supported options:
```
./run.py --help
```

For now, it is assumed that `-a x64_64` (default) is native compilation, in which case an actual run 
is triggered to check and benchmark the implementation.
Other arch, e.g. `-a arm64` is considered cross-compilation and stops after llvm-mca.
