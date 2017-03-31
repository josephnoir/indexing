
// For optimizations regarding bank conflicts, look at:
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// TODO: Get efficient scan to work with arbitrary n > 0
// scan consisting of upsweep, null_last and downsweep does not work!!!


kernel void upsweep(global uint* config, global uint* data) {
  uint n = config[0];
  uint d = config[1];
  uint idx = get_global_id(0);
  uint bound = UINT_MAX;
  if (idx > 0)
    bound /= idx;
  uint foo = 1u << d;
  uint inc = foo << 1;
  uint k = idx * inc;
  if (idx != 0 && (inc > bound))
    k = n;
  uint to = k + inc - 1;
  if (k <= n - 1 && to <= n - 1) {
    uint left = k + foo - 1;
    uint right = k + inc - 1;
    data[k + inc - 1] = data[k + foo - 1] + data[k + inc - 1]; // not sure if last one uses inc or foo + 1
  }
}

kernel void null_last(global uint* config, global uint* data) {
  uint n = config[0];
  data[n - 1] = 0;
}

kernel void downsweep(global uint* config, global uint* data) {
  uint n = config[0];
  uint d = config[1];
  uint idx = get_global_id(0);
  uint bound = UINT_MAX;
  if (idx > 0)
    bound /= idx;
  uint foo = 1u << d;
  uint inc = foo << 1;
  uint k = idx * inc;
  if (idx != 0 && (inc > bound))
    k = n;
  uint to = k + inc - 1;
  if (k <= n - 1 && to <= n - 1) {
    uint tmp = data[k + foo - 1];
    data[k + foo - 1] = data[k + inc - 1];
    data[k + inc - 1] = tmp + data[k + inc - 1];
  }
}

kernel void lazy_scan(global uint* in, global uint* out, private uint len) {
  uint idx = get_global_id(0);
  if (idx == 0) {
    out[0] = 0;
    for (uint i = 1; i < len; ++i)
      out[i] = out[i - 1] + in[i - 1];
  }
}
