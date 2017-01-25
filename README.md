# Indexing

Experimenting with indexing on GPUs, see this [paper for WAH](http://alumni.cs.ucr.edu/~mvlachos/pubs/netfli_gpu.pdf). Includes implementation for CPU (C++) and GPU (OpenCL)--CPU first to understand the algorithm, than move to GPU.

Eventually, more indexing algorithms should implemented (EWAH, Roaring, ...). Further, comparison with and, if performant, integration into [VAST](vast.io) would be nice.

Currently an experiment with a focus on working code, optimizations will hopefully follow.

## Build

Use the `configure` script to run cmake and pass the path to yout CAF build directory (if not installed).

```
$ ./configure [--with-caf=CAF_BUILD_DIR]
$ make
```

*Note:* `gpu` program currently configured to load cl file from include directory, execute from project root using `./build/bin/gpu`. Number of indexed values should fit into a working group (simple algorithms currently imposed that limit), the gpu main file uses the value `amount` to determin how many values are created. (Still needs a lot of work).

## Requirements

* CAF build with the OpenCL submodule
* C++ compiler

