typedef struct GPUConstants {
     int numRadices;
     int numBlocks;
     int numGroupsPerBlock;
     int R;
     int numThreadsPerGroup;
     int numElementsPerGroup;
	 int numRadicesPerBlock;
     int bitMask;
     int L;
     int numThreadsPerBlock;
     int numTotalElements;
}Constants;

kernel void zeroes(global uint* counters) {
  counters[get_global_id(0)] = 0;
}

inline void PrefixLocal(__local uint* inout, int p_length, int numThreads){
  __private uint glocalID = get_local_id(0);
  __private int inc = 2;
  while(inc <= p_length){
    for(int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + inc) < p_length ; i+= numThreads*inc){
        inout[i + (inc>>1)] = inout[i] + inout[i + (inc>>1)];
    }
    inc = inc <<1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  inout[p_length-1] = 0;
  barrier(CLK_GLOBAL_MEM_FENCE);
  while(inc >=2){
    for (int i = ((inc>>1) - 1) + (glocalID * inc)  ; (i + (inc>>1)) <= p_length ; i+= numThreads*inc)
    {
      uint tmp = inout[i + (inc >>1)];
      inout[i + (inc >>1)] = inout[i] + inout[i + (inc >>1 )];
      inout[i] = tmp;
    }
    inc = inc>>1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}


__kernel void SetupAndCount(__global  uint* cellIdIn,
    							          __global volatile uint* counters,
							              Constants dConst,
							              uint bitOffset)
{
	__private uint gLocalId = get_local_id(0);
	__private uint gBlockId = get_group_id(0);
	__private uint threadGroup = gLocalId / dConst.R;
  __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
  __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;
  __private uint idx = actBlock + actGroup + gLocalId % dConst.R;
  __private int boarder = actBlock +actGroup + dConst.numElementsPerGroup;
  boarder = (boarder > dConst.numTotalElements)? dConst.numTotalElements : boarder;
  __private uint countersPerRadix = dConst.numBlocks * dConst.numGroupsPerBlock;
  __private uint counterGroupOffset = gBlockId * dConst.numGroupsPerBlock;
	for(;idx < boarder; idx += dConst.numThreadsPerGroup){
		__private uint actRadix = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
   for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
      if(gLocalId % dConst.R == tmpIdx){
          //counters[ (actRadix * countersPerRadix)  +counterGroupOffset+ threadGroup ]++;
          atomic_inc(&counters[(actRadix * countersPerRadix)  +counterGroupOffset+ threadGroup ]);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
	}
}

__kernel void SumIt(__global uint* cellIdIn,
                    __global volatile uint* counters,
                    __global uint* radixPrefixes,
                    __local uint* groupcnt,
                    Constants dConst,
                    uint bitOffset)
{
	__private uint globalId = get_global_id(0);
	__private uint gLocalId = get_local_id(0);
	__private uint gBlockId = get_group_id(0);
  __private uint countersPerRadix = dConst.numBlocks * dConst.numGroupsPerBlock;
	__private uint actRadix = dConst.numRadicesPerBlock * gBlockId;
  for(int i = 0; i < dConst.numRadicesPerBlock; i++){
    int numIter = 0;
    uint boarder = ((actRadix+1) * countersPerRadix);
    for(int j = (actRadix * countersPerRadix) + gLocalId ; j < boarder; j+= dConst.numThreadsPerBlock){
      groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++] = counters[j];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    PrefixLocal(groupcnt, countersPerRadix ,dConst.numThreadsPerBlock );
    barrier(CLK_LOCAL_MEM_FENCE);
    if(gLocalId == 1){
      radixPrefixes[actRadix] = groupcnt[(countersPerRadix) -1] + counters[((actRadix+1) * countersPerRadix)-1];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    numIter = 0;
    for(int j = (actRadix * countersPerRadix) + gLocalId ; j < ((actRadix+1) * countersPerRadix); j+= dConst.numThreadsPerBlock){
      counters[j] = groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
    actRadix++;
  }
}



__kernel void ReorderingKeysOnly(__global uint* cellIdIn, 
                            __global uint* cellIdOut,
                            __global uint* counters,
                            __global uint* radixPrefixes,
                            __local uint* localCounters,
                            __local uint* localPrefix,
                             Constants dConst,
                            uint bitOffset)
{
  int globalId = get_global_id(0);
  __private uint gLocalId = get_local_id(0);
  __private uint gBlockId = get_group_id(0);
  __private uint threadGroup= gLocalId / dConst.R;
  __private uint actRadix = dConst.numRadicesPerBlock * gBlockId;
  __private uint countersPerRadix = dConst.numGroupsPerBlock * dConst.numBlocks;
  __private int radixCounterOffset = actRadix * countersPerRadix;
  __private  uint blockidx ;
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
    for(uint   blockidx = gLocalId;
                blockidx < countersPerRadix ;
                 blockidx+= dConst.numThreadsPerBlock){
      localCounters[ i* countersPerRadix  +  blockidx] = counters[radixCounterOffset +  i* countersPerRadix  +  blockidx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int i = gLocalId ; i< dConst.numRadices ; i+= dConst.numThreadsPerBlock){
    localPrefix[i] = radixPrefixes[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  PrefixLocal(localPrefix, dConst.numRadices, dConst.numThreadsPerBlock);
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
    int numIter = 0;
    for(int j =  gLocalId ; j < countersPerRadix; j+= dConst.numThreadsPerBlock){
      localCounters[ i* countersPerRadix  +  j] += localPrefix[actRadix+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
    for(uint   blockidx =gLocalId;
                blockidx <  countersPerRadix;
                 blockidx+= dConst.numThreadsPerBlock){
      counters[radixCounterOffset +  i* countersPerRadix  +  blockidx] = localCounters[ i* countersPerRadix  +  blockidx];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
  __private int actBlockCounter = gBlockId * dConst.numGroupsPerBlock  ;

  __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;

  __private  uint idx = actBlock + actGroup +  gLocalId % dConst.R;
  int boundary = actBlock+ actGroup +  dConst.numElementsPerGroup;
  boundary = (actBlock+ actGroup +  dConst.numElementsPerGroup < dConst.numTotalElements)? boundary : dConst.numTotalElements;
  for(;idx <   boundary ; idx += dConst.numThreadsPerGroup){
    uint tmpRdx = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
    for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
      if(gLocalId % dConst.R == tmpIdx){
          cellIdOut[counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]] = cellIdIn[idx];
          atomic_inc(&counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}


__kernel void ReorderingKeyValue(__global uint* cellIdIn, 
                            __global uint* cellIdOut,
                            __global uint* valueIn,
                            __global uint* valueOut,
                            __global uint* counters,
                            __global uint* radixPrefixes,
                            __local uint* localCounters,
                            __local uint* localPrefix,
                             Constants dConst,
                            uint bitOffset)
{
  int globalId = get_global_id(0);
    __private uint gLocalId = get_local_id(0);
    __private uint gBlockId = get_group_id(0);
    __private uint threadGroup= gLocalId / dConst.R;
    __private uint actRadix = dConst.numRadicesPerBlock * gBlockId;
    __private uint countersPerRadix = dConst.numGroupsPerBlock * dConst.numBlocks;
    __private int radixCounterOffset = actRadix * countersPerRadix;
    // erst abschließen der radix summierung
    __private  uint blockidx ; 
    for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;
                    blockidx <  countersPerRadix ; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            localCounters[ i* countersPerRadix  +  blockidx] = counters[radixCounterOffset +  i* countersPerRadix  +  blockidx];
        }
    }

    // Read radix prefixes to localMemory
    for(int i = gLocalId ; i< dConst.numRadices ; i+= dConst.numThreadsPerBlock){
        localPrefix[i] = radixPrefixes[i];
    }

    // Präfixsumme über die RadixCounter bilden.
    barrier(CLK_GLOBAL_MEM_FENCE);
    PrefixLocal(localPrefix, dConst.numRadices, dConst.numThreadsPerBlock);
    barrier(CLK_GLOBAL_MEM_FENCE);
 



    // Die Präfixsumme des Radixe auf alle subcounter der radixes addieren
  for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        int numIter = 0;
        //for(int j = ((actRadix+i) * dConst.numBlocks * dConst.numGroupsPerBlock) + localId ; j < ((dConst.numRadicesPerBlock * gBlockId + 1) * dConst.numBlocks * dConst.numGroupsPerBlock); j+= dConst.numThreadsPerBlock){
        for(int j =  gLocalId ; j < countersPerRadix; j+= dConst.numThreadsPerBlock){
            //groupcnt[gLocalId + dConst.numThreadsPerBlock * numIter++] = counters[j];
            //if(gLocalId == 0 && gBlockId ==0 && i==0 && j == gLocalId )
            localCounters[ i* countersPerRadix  +  j] += localPrefix[actRadix+i];

        }
       // barrier(CLK_GLOBAL_MEM_FENCE);
    }
    



    // Zurückschreiben der Radixe mit entsprechedem offset.
    for(int i = 0 ; i< dConst.numRadicesPerBlock ; i++){
        for(uint   blockidx =gLocalId;  
                    blockidx <  countersPerRadix; 
                     blockidx+= dConst.numThreadsPerBlock){
            // The Num_Groups counters of the radix are read from global memory to shared memory.
            // Jeder Thread liest die Counter basierend auf der groupId aus
            counters[radixCounterOffset +  i* countersPerRadix  +  blockidx] = localCounters[ i* countersPerRadix  +  blockidx];
        }
    }



    barrier(CLK_GLOBAL_MEM_FENCE);



    __private int actBlock = gBlockId * dConst.numGroupsPerBlock * dConst.numElementsPerGroup ;
    __private int actBlockCounter = gBlockId * dConst.numGroupsPerBlock  ;

    __private int actGroup = (gLocalId / dConst.R ) * dConst.numElementsPerGroup;

    __private  uint idx = actBlock + actGroup +  gLocalId % dConst.R;
    int boundary = actBlock+ actGroup +  dConst.numElementsPerGroup;
    boundary = (actBlock+ actGroup +  dConst.numElementsPerGroup < dConst.numTotalElements)? boundary : dConst.numTotalElements;
    for(;idx <   boundary ; idx += dConst.numThreadsPerGroup){
        uint tmpRdx = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        //uint outputIdx = counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++;
        for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(gLocalId % dConst.R == tmpIdx){

                cellIdOut[counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]] = cellIdIn[idx];
                valueOut[counters[tmpRdx * countersPerRadix + actBlockCounter+ threadGroup]++] = valueIn[idx];
                //cellIdOut[idx] = gLocalId+1;

            }
            barrier(CLK_GLOBAL_MEM_FENCE);


        }

    }

/*

    __private  uint idx = (gBlockId * dConst.numGroupsPerBlock + gLocalId / dConst.R) * dConst.numElementsPerGroup + gLocalId % dConst.R;

    for(;idx < (gBlockId * dConst.numGroupsPerBlock + threadGroup+1) * dConst.numElementsPerGroup; idx += dConst.numThreadsPerGroup){
        uint actRadix = (cellIdIn[idx] >> bitOffset) & dConst.bitMask;
        //uint outputIdx = counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++;
        for(uint tmpIdx = 0 ; tmpIdx < dConst.R; tmpIdx++){
            if(gLocalId % dConst.R == tmpIdx){
                cellIdOut[counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]] = cellIdIn[idx];
                valueOut[counters[actRadix * countersPerRadix + gBlockId * dConst.numGroupsPerBlock + threadGroup]++] = valueIn[idx];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);


        }

    }
*/

}


