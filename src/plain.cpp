#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "caf/all.hpp"

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

using namespace std;
using namespace caf;

namespace {

constexpr const char* kernel_name = "kernel_wah_index";
constexpr const char* kernel_file = "kernel_wah_bitindex.cl";

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  string device_name = "GeForce GT 650M";
  uint32_t batch_size = 1023;
  uint32_t jobs = 1;
  uint32_t work_groups = 1;
  bool print_results;
  config() {
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M)")
    .add(batch_size, "batch-size,b", "values indexed in one batch (1023)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(work_groups, "work-groups,w", "Use work-groups")
    .add(jobs, "jobs,j", "jobs sent to GPU (1)");
  }
};


void eval_err(cl_int err, string msg = "") {
  if (err != CL_SUCCESS) {
    cout << "[!!!] " << msg << err << endl;
    exit(err);
  }
}

} // namespace <anonymous>

void caf_main(actor_system&, const config& cfg) {

  vector<uint32_t> values;
  if (cfg.filename.empty()) {
    values = {10,  7, 22,  6,  7,  1,  9, 42,  2,  5,
              13,  3,  2,  1,  0,  1, 18, 18,  3, 13,
               5,  9,  0,  3,  2, 19,  5, 23, 22, 10,
               6, 22};
  } else {
    ifstream source{cfg.filename, std::ios::in};
    uint32_t next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }

  // read cl kernel file
  auto filename = string("./include/") + kernel_file;
  ifstream read_source{filename, std::ios::in};
  string source_contents;
  if (read_source) {
      read_source.seekg(0, std::ios::end);
      source_contents.resize(read_source.tellg());
      read_source.seekg(0, std::ios::beg);
      read_source.read(&source_contents[0], source_contents.size());
      read_source.close();
  } else {
      cout << strerror(errno) << "[!!!] " << endl;
      return;
  }

  cl_int err{0};

  // find up to two available platforms
  uint32_t num_platforms;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  vector<cl_platform_id> platform_ids(num_platforms);
  err = clGetPlatformIDs(platform_ids.size(), platform_ids.data(),
                         &num_platforms);
  eval_err(err);

  // find gpu devices on our platform
  int platform_used = 0;
  uint32_t num_gpu_devices = 0;
  err = clGetDeviceIDs(platform_ids[platform_used], CL_DEVICE_TYPE_GPU, 0,
                       nullptr, &num_gpu_devices);
  eval_err(err);
  vector<cl_device_id> gpu_devices(num_gpu_devices);
  err = clGetDeviceIDs(platform_ids[platform_used], CL_DEVICE_TYPE_GPU,
                       num_gpu_devices, gpu_devices.data(), nullptr);
  eval_err(err);

  // choose device
  int device_used{0};
  bool found = false;
  if (!cfg.device_name.empty()) {
    for (uint32_t i = 0; i < num_gpu_devices; ++i) {
      size_t return_size;
      err = clGetDeviceInfo(gpu_devices[i], CL_DEVICE_NAME, 0, nullptr,
                            &return_size);
      vector<char> name(return_size);
      err = clGetDeviceInfo(gpu_devices[i], CL_DEVICE_NAME, return_size,
                            name.data(), &return_size);
      string as_string(name.data());
      if (as_string == cfg.device_name) {
        device_used = i;
        found = true;
        break;
      }
    }
    if (!found) {
      cout << "Device " << cfg.device_name << " not found." << endl;
      return;
    }
  } else {
    size_t return_size;
    err = clGetDeviceInfo(gpu_devices[device_used], CL_DEVICE_NAME, 0, nullptr,
                          &return_size);
    vector<char> name(return_size);
    err = clGetDeviceInfo(gpu_devices[device_used], CL_DEVICE_NAME, return_size,
                          name.data(), &return_size);
    cout << "Using '" << string(name.data()) << "'" << endl;
  }
  cl_device_id device_id_used = gpu_devices[device_used];
  size_t max_work_group_size;
  err = clGetDeviceInfo(device_id_used, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(size_t), &max_work_group_size, NULL);

  // create a context
  auto pfn_notify = [](const char *errinfo, const void *, size_t, void *) {
      std::cerr << "\n##### Error message via pfn_notify #####\n"
                << string(errinfo)
                << "\n########################################";
  };
  cl_context context = clCreateContext(0, 1, &device_id_used, pfn_notify,
                                       nullptr, &err);
  eval_err(err);


  // create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id_used,
                                                    CL_QUEUE_PROFILING_ENABLE,
                                                    &err);
  eval_err(err);

  // create program object from kernel source
  const char* strings = source_contents.c_str();
  size_t      lengths = source_contents.size();
  cl_program program = clCreateProgramWithSource(context, 1, &strings,
                                                 &lengths, &err);
  eval_err(err);

  // build programm from program object
  err = clBuildProgram(program, 0, nullptr, "", nullptr,
                       nullptr);
  eval_err(err);


  // create kernel
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  eval_err(err);

  auto batch_size = min(cfg.batch_size, static_cast<uint32_t>(values.size()));
  auto wg_size = min(batch_size, static_cast<uint32_t>(max_work_group_size));
  auto wg_num = cfg.work_groups;
  auto gl_size = wg_size * wg_num; // must be multiple of wg_size
  auto processed = 0;
  auto remaining = static_cast<uint32_t>(values.size());
  auto in_out_flags = CL_MEM_READ_WRITE;
  auto out_flags    = CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY;
  auto buf_flags    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
  auto blocking = CL_FALSE;
   auto start = chrono::high_resolution_clock::now();
  while (remaining > 0) {
    auto arg_size = min(remaining, wg_num * wg_size);
    auto arg_bytes = arg_size * sizeof(uint32_t);
    vector<uint32_t> input{
      begin(values) + processed,
      begin(values) + processed + arg_size
      //make_move_iterator(begin(values) + processed),
      //make_move_iterator(begin(values) + processed + arg_size)
    };
    processed += arg_size;
    remaining -= arg_size;
    auto full = arg_size / wg_size;
    auto partial = arg_size - (full * wg_size);
    vector<uint32_t> config(2 + (full * 2));
    config[0] = arg_size;
    config[1] = full;
    for (uint32_t i = 0; i < full; ++i) {
      config[2 * i + 2] = wg_size;
      config[2 * i + 3] = wg_size * 2;
    }
    if (partial > 0) {
      config[1] += 1;
      config.emplace_back(partial);
      config.emplace_back(partial * 2);
    }
    auto cfg_bytes = config.size() * sizeof(uint32_t);

    vector<cl_event> eq_events;

    // create input buffers
    cl_mem buf_config = clCreateBuffer(context, in_out_flags, cfg_bytes,
                                       nullptr, &err);
    eval_err(err);
    cl_mem buf_input = clCreateBuffer(context, in_out_flags, arg_bytes,
                                      nullptr, &err);
    eval_err(err);
    cl_mem buf_index = clCreateBuffer(context, out_flags, arg_bytes * 2,
                                      nullptr, &err);
    eval_err(err);
    cl_mem buf_offsets = clCreateBuffer(context, out_flags, arg_bytes,
                                        nullptr, &err);
    eval_err(err);
    cl_mem buf_rids = clCreateBuffer(context, buf_flags, arg_bytes,
                                     nullptr, &err);
    eval_err(err);
    cl_mem buf_chids = clCreateBuffer(context, buf_flags, arg_bytes,
                                      nullptr, &err);
    eval_err(err);
    cl_mem buf_lits = clCreateBuffer(context, buf_flags, arg_bytes,
                                     nullptr, &err);
    eval_err(err);

    // copy data to GPU
    cl_event config_cpy;
    err = clEnqueueWriteBuffer(cmd_queue, buf_config, blocking, 0,
                               cfg_bytes, config.data(), 0,
                               nullptr, &config_cpy);
    eval_err(err);
    eq_events.push_back(config_cpy);
    cl_event input_copy;
    err = clEnqueueWriteBuffer(cmd_queue, buf_input, blocking, 0,
                               arg_bytes, input.data(), 0,
                               nullptr, &input_copy);
    eval_err(err);
    eq_events.push_back(input_copy);

    // set arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &buf_config);
    eval_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &buf_input);
    eval_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &buf_index);
    eval_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &buf_offsets);
    eval_err(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &buf_rids);
    eval_err(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &buf_chids);
    eval_err(err);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*) &buf_lits);
    eval_err(err);

    // enqueue kernel
    vector<size_t> global_work{gl_size};
    vector<size_t> local_work{wg_size};
    cl_event kernel_exec;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr,
                                 global_work.data(), local_work.data(),
                                 eq_events.size(), eq_events.data(),
                                 &kernel_exec);
    eval_err(err);

    // get results from gpu
    vector<uint32_t> keys(arg_size);
    vector<uint32_t> index(arg_size * 2);
    vector<uint32_t> offsets(arg_size);
    err = clEnqueueReadBuffer(cmd_queue, buf_config, blocking, 0,
                              cfg_bytes, config.data(), 1,
                              &kernel_exec, nullptr);
    eval_err(err);
    err = clEnqueueReadBuffer(cmd_queue, buf_input, blocking, 0,
                              arg_bytes, keys.data(), 1,
                              &kernel_exec, nullptr);
    eval_err(err);
    err = clEnqueueReadBuffer(cmd_queue, buf_index, blocking, 0,
                              arg_bytes * 2, index.data(), 1,
                              &kernel_exec, nullptr);
    eval_err(err);
    err = clEnqueueReadBuffer(cmd_queue, buf_offsets, blocking, 0,
                              arg_bytes, offsets.data(), 1,
                              &kernel_exec, nullptr);
    eval_err(err);
    clFinish(cmd_queue);

    for (auto& ev : eq_events)
      clReleaseEvent(ev);
    clReleaseEvent(kernel_exec);

    clReleaseMemObject(buf_config);
    clReleaseMemObject(buf_input);
    clReleaseMemObject(buf_index);
    clReleaseMemObject(buf_offsets);
    clReleaseMemObject(buf_rids);
    clReleaseMemObject(buf_chids);
    clReleaseMemObject(buf_lits);
  }
  auto stop = chrono::high_resolution_clock::now();
  // clean up
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);
  cout << "Time: '"
       << chrono::duration_cast<chrono::milliseconds>(stop - start).count()
       << "' ms" << endl;
}

CAF_MAIN()
