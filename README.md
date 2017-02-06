# Indexing

Experimenting with indexing on GPUs, see this [paper for WAH](http://alumni.cs.ucr.edu/~mvlachos/pubs/netfli_gpu.pdf). Includes implementation for CPU (C++), GPU (OpenCL)--CPU first to understand the algorithm, than move to GPU--and a CPU implementation that uses VAST for comparison.

Eventually, more indexing algorithms should implemented (EWAH, Roaring, ...). Further, comparison with and, if performant, integration into [VAST](vast.io) would be nice.

Currently an experiment with a focus on working code, optimizations will follow.

## Build

Use the `configure` script to run cmake and pass the path to yout CAF build directory (if not installed).

```
$ ./configure [--with-caf=CAF_BUILD_DIR] [--with-vast=VAST_BUILD_DIR]
$ make
```

This builds the following programs, each has a `--help` option that lists all command line arguments:

* `cpu`: The GPU indexing algorithm implemented with C++, not efficient of scalable
* `gpu`: Implementation on the GPU with OpenCL, uses one thread per value, can use multiple work-groups concurrently
* `vst`: Index using VAST
* `generate`: Generate test data, per default 1 GB on 32 bit unsigned integers in the range of 0 to 1023

**Note:** `gpu` program currently configured to load the kernel from a `.cl` file in include directory when executed from project root using `./build/bin/gpu`.

## Requirements

* [CAF](https://github.com/actor-framework/actor-framework) (build with the [OpenCL submodule](https://github.com/actor-framework/opencl))
* [VAST](https://github.com/vast-io) (requires clang >= 3.5 or gcc >= 6)
* C++ 14 compiler
* OpenCL (version 1.1 or 1.2)
