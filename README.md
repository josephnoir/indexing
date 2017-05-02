# Indexing

Experimenting with indexing on GPUs, see this [paper for WAH](http://alumni.cs.ucr.edu/~mvlachos/pubs/netfli_gpu.pdf). Includes implementation for CPU (C++), GPU (OpenCL), and a CPU implementation that uses VAST for comparison.

Eventually, more indexing algorithms should implemented (EWAH, Roaring, ...). Further, comparison with and, if performant, integration into [VAST](vast.io) would be nice.

Indexing seems to work mostly, but it is still very slow--most of all the radix sort.

## Build

Use the `configure` script to run cmake and pass the path to yout CAF build directory (if not installed).

```
$ ./configure [--with-caf=CAF_BUILD_DIR] [--with-vast=VAST_BUILD_DIR]
$ make
```

This builds the following programs, each has a `--help` option that lists all command line arguments:

* `cpu`: The GPU indexing algorithm implemented with C++, not efficient of scalable
* `gpu`: Implementation on the GPU with OpenCL, uses one thread per value, can use multiple work groups concurrently
* `vst`: Index using VAST
* `phases`: Indexing implementation that tries to keep data on the GPU using multiple kernel invocations
* `generate`: Generate test data, per default 1 GB on 32 bit unsigned integers in the range of 0 to max of `uint16_t`
* More programs to develop specific functionality, such as scan

**Note:** Kernels are currently read from `.cl` files in the include directory when executed from project root using `./build/bin/$PROGRAM`.

## Requirements

* [CAF](https://github.com/actor-framework/actor-framework) (build with the [OpenCL submodule](https://github.com/actor-framework/opencl))
* [VAST](https://github.com/vast-io) (requires clang >= 3.5 or gcc >= 6)
* C++ 14 compiler
* OpenCL (version 1.1 or 1.2)

Required OpenCL submodule branch is [topic/multi_phase_kernels](https://github.com/actor-framework/opencl/tree/topic/multi_phase_kernels).


## TODOs

- [x] Segmented Scan
- [ ] Scan for data of arbitrary size (currently limited to ~1 Million)
- [x] Segmented Scan for arbitrary size
- [ ] Use segmented scan for indexing
- [ ] Stream compaction using scan + move kernels
- [ ] More work per work item in `fuse fill literals` kernel
