/******************************************************************************
 * function: fuse_fill_literals                                               *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


/**
 * High level function:
 * size_t fuse_fill_literals(global uint* chids, global uint* lits,
 *                           global uint* index, size_t li, size_t work_size);
 *
 */

/*
size_t fuse_fill_literals(global uint* chids, global uint* lits,
                          global uint* index, size_t li, size_t work_size) {
  local uint markers [WORK_GROUP_SIZE * 2];
  local uint position[WORK_GROUP_SIZE * 2];
  volatile local int len;
  len = 0;
  uint a =  2 * li;
  uint b = (2 * li) + 1;
  index[a] = chids[li];
  index[b] = lits[li];
  barrier(FENCE_TYPE);
  // stream compaction
  markers[a] = index[a] != 0; // ? 1 : 0;
  markers[b] = index[b] != 0; // ? 1 : 0;
  position[a] = 0;
  position[b] = 0;
  barrier(FENCE_TYPE);
  // should be a parallel scan
  if (li == 0) {
    for (uint i = 1; i < WORK_GROUP_SIZE * 2; ++i) {
      position[i] = position[i - 1] + markers[i - 1];
    }
  }
  // end parallel scan
  uint tmp_a = index[a];
  uint tmp_b = index[b];
  barrier(FENCE_TYPE);
  if (li < work_size) {
    if (markers[a] == 1) {
      index[position[a]] = tmp_a;
      atomic_add(&len, 1);
      //atomic_max(&len, position[a] + 1);
    }
    if (markers[b] == 1) {
      index[position[b]] = tmp_b;
      atomic_add(&len, 1);
      //atomic_max(&len, position[b] + 1);
    }
  }
  // end stream compaction
  barrier(FENCE_TYPE); // <-- should there be a barrier here?
  return len;
}
*/
