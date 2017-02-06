#include <vector>
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

} // namespace <anonymous>


class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  string device_name = "GeForce GTX 780M";
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
  if (err != CL_SUCCESS)
    cout << "[!!!] " << msg << err << endl;
  else
    cout << "DONE" << endl;
}

void caf_main(actor_system& system, const config& cfg) {


  vector<uint32_t> values;
  if (cfg.filename.empty()) {
    values = {10,  7, 22,  6,  7,  1,  9, 42,  2,  5,
              13,  3,  2,  1,  0,  1, 18, 18,  3, 13,
               5,  9,  0,  3,  2, 19,  5, 23, 22, 10,
               6, 22};
  } else {
    cout << "Reading data from '" << cfg.filename << "' ... " << flush;
    ifstream source{cfg.filename, std::ios::in};
    uint32_t next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  cout << "'" << values.size() << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is '" << bound << "'." << endl;

  // read cl kernel file
  auto filename = string("./include/") + kernel_file;
  cout << "Reading source from '" << filename << "' ... " << flush;
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
  cout << "DONE" << endl;

  // #### OLD PROGRAM ####

  cl_int err{0};

  // find up to two available platforms
  cout << "Getting platform id(s) ..." << flush;
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  vector<cl_platform_id> platform_ids(num_platforms);
  err = clGetPlatformIDs(platform_ids.size(), platform_ids.data(),
                         &num_platforms);
  eval_err(err); 

  // find gpu devices on our platform
  int platform_used = 0;
  cl_uint num_gpu_devices = 0;
  err = clGetDeviceIDs(platform_ids[platform_used], CL_DEVICE_TYPE_GPU, 0,
                       nullptr, &num_gpu_devices);
  if (err != CL_SUCCESS) {
      cout << "[!!!] Error getting number of gpu devices for platform '"
           << platform_ids[platform_used] << "'." << endl;
      return;
  }
  cl_device_id gpu_devices[num_gpu_devices];
  err = clGetDeviceIDs(platform_ids[platform_used], CL_DEVICE_TYPE_GPU,
                       num_gpu_devices, gpu_devices, nullptr);
  if (err != CL_SUCCESS) {
      cout << "[!!!] Error getting CL_DEVICE_TYPE_GPU for platform '"
           << platform_ids[platform_used] << "'." << endl;
      return;
  }

  // choose device
  int device_used{0};
  bool found = false;
  if (cfg.device_name.empty()) {
    for (cl_uint i = 0; i < num_gpu_devices; ++i) {
      size_t return_size;
      err = clGetPlatformInfo(gpu_devices[i], CL_DEVICE_NAME, 0, nullptr, 
                              &return_size);
      vector<char> name(return_size);
      err = clGetPlatformInfo(gpu_devices[i], CL_DEVICE_NAME, return_size,
                              name.data(), 
                              &return_size);
      if (string(name.data()) == cfg.device_name) {
        device_used = i;
        found = true;
        break;
      }
    }
  }

  if (!found) {
    cout << "Device " << cfg.device_name << " not found." << endl;
    return;
  }
  cl_device_id device_id_used = gpu_devices[device_used];
  cout << "Using device: '" << device_id_used << "'." << endl;

  // create a context
  auto pfn_notify = [](const char *errinfo,
                       const void *,
                       size_t,
                       void *) {
      std::cerr << "\n##### Error message via pfn_notify #####\n"
                << string(errinfo)
                << "\n########################################";
  };
  cout << "Creating context ... " << flush;
  cl_context context = clCreateContext(0, 1, &device_id_used, pfn_notify,
                                       nullptr, &err);
  eval_err(err); 


  // create a command queue
  cout << "Creating command queue ... " << flush;
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id_used,
                                                    CL_QUEUE_PROFILING_ENABLE,
                                                    &err);
  eval_err(err);

  // create program object from kernel source
  cout << "Creating program object from source ... " << flush; 
  size_t kernel_source_length = strlen(source_contents);
  cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_ptr,
                                                 &source_contents.size(), &err);
  eval_err(err);

  // build programm from program object
  cout << "Building program from program object ... " << flush;
  err = clBuildProgram(program, 0, nullptr, "-cl-kernel-arg-info", nullptr,
                       nullptr);
  eval_err(err);


  // create kernel
  cout << "Creating kernel object ... " << flush;
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  eval_err(err); 

  // get kernel info
  cout << "Getting kernel information" << endl;
  size_t kernel_info_size;
  std::vector<char> kernel_info_buffer;
  cout << "Getting CL_KERNEL_FUNCTION_NAME ... " << flush;
  err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, 
                        &kernel_info_size);
  kernel_info_buffer.resize(kernel_info_size);
  err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_info_size,
                        kernel_info_buffer.data(), nullptr);
  eval_err(err);
  cout << "> " << kernel_info_buffer.data() << endl;
  kernel_info_buffer.clear();
  cout << "Getting CL_KERNEL_ATTRIBUTES ... " << flush;
  err = clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, 0, nullptr,
                        &kernel_info_size);
  kernel_info_buffer.resize(kernel_info_size);
  err = clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, kernel_info_size,
                        kernel_info_buffer.data(), nullptr);
  eval_err(err);
  cout << "> " << kernel_info_buffer.data() << endl;
  cout << "Getting CL_KERNEL_NUM_ARGS ... " << flush;
  cl_uint number_of_arguments;
  err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint),
                        &number_of_arguments, nullptr);
  eval_err(err);
  cout << "> " << number_of_arguments << endl;

  // get arg information
  for (int i = 0; i < number_of_arguments; ++i) {
      cout << "Getting CL_KERNEL_ARG_ADDRESS_QUALIFIER for arg " << i << " ... "
           << flush;
      cl_kernel_arg_address_qualifier qualifier;
      err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                               sizeof(cl_kernel_arg_address_qualifier),
                               &qualifier, nullptr);
      eval_err(err);
      switch(qualifier) {
          case (CL_KERNEL_ARG_ADDRESS_GLOBAL):
              cout << "> global" << endl;
              break;
          case (CL_KERNEL_ARG_ADDRESS_LOCAL):
              cout << "> local" << endl;
              break;
          case (CL_KERNEL_ARG_ADDRESS_CONSTANT):
              cout << "> constant" << endl;
              break;
          case (CL_KERNEL_ARG_ADDRESS_PRIVATE):
              cout << "> private" << endl;
              break;
      }
  }
  

  // create input and output arrays
  vector<int> input(buffer_size);
  int n{static_cast<int>(input.capacity())};
  generate(begin(input), end(input), [&]{ return --n; });

  vector<cl_event> eq_events;

  // create and publish buffer
  // cout << "Creating buffer 'buffer' ... " << flush;
  // cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * buffer_size, input.data(), &err);
  // eval_err(err); 
  // cout << "Creating buffer 'result' ... " << flush;
  // cl_mem result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * result_size, nullptr, &err);
  // eval_err(err); 
  cout << "Creating buffer 'buffer' ... " << flush;
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * buffer_size, nullptr, &err);
  eval_err(err); 
  cout << "Creating buffer 'result' ... " << flush;
  cl_mem result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * result_size, nullptr, &err);
  eval_err(err); 

  // copy data to GPU
  cl_event cpy_data;
  err = clEnqueueWriteBuffer(cmd_queue, buffer, CL_FALSE, 0, sizeof(int) * buffer_size, input.data(), 0, nullptr, &cpy_data);
  eq_events.push_back(cpy_data);


  cl_event event;

  // set arguments
  cout << "Setting kernel argument 0 ... " << flush;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &buffer);
  eval_err(err);

  cout << "Setting kernel argument 1 ... " << flush;
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &result);
  eval_err(err); 
  clFinish(cmd_queue);
  

  // enqueue kernel
  cout << "Enqueueing kernel to command queue ... " << flush;
  vector<size_t> global_work_size{global_size};
  vector<size_t> local_work_size{local_size};

  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 
                               global_work_size.size(), 
                               nullptr, 
                               global_work_size.data(), 
                               local_work_size.data(), 
                               eq_events.size(), 
                               eq_events.data(),
                               &event);
  clReleaseEvent(event);
  eval_err(err); 
  clFinish(cmd_queue);
  

  // get results from gpu
  cout << "Reading results from results ... " << flush;
  vector<int> output(result_size);
  err = clEnqueueReadBuffer(cmd_queue, result, CL_TRUE, 0, sizeof(int) * result_size, output.data(), 0, nullptr, &event);
  clReleaseEvent(event);
  eval_err(err); 
  
  cout << "results: ";
  for (auto& e : output) {
      cout << e << " ";
  }
  cout << endl;

  // clean up
  cout << "Releasing memory ... " << flush;
  if (kernel)    clReleaseKernel(kernel);
  if (program)   clReleaseProgram(program);
  if (cmd_queue) clReleaseCommandQueue(cmd_queue);
  
  if (buffer)    clReleaseMemObject(buffer);
  if (result)   clReleaseMemObject(result);

  if (context)   clReleaseContext(context);
  cout << "DONE" << endl;
}

CAF_MAIN()
