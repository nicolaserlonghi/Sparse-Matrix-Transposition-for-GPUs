#include <prefixSum.cuh>

const int NUM_THREADS = 128;


// Function prototypes
__global__
void uniformAdd(int *outputArray, int numElements, int *INCR);
__global__
void blockPrefixSum(int *g_idata, int *g_odata, int n, int *SUM);



int manageMemoryForPrefixSum(int numElements) {
    int blocksPerGridL1 = 1 + (numElements - 1) / (NUM_THREADS * 2);
    int blocksPerGridL2 = 1 + blocksPerGridL1 / (NUM_THREADS * 2);
    int blocksPerGridL3 = 1 + blocksPerGridL2 / (NUM_THREADS * 2);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    int clean = 1;
    if (blocksPerGridL1 != 1 && blocksPerGridL2 == 1) {
        double occupancy = ((blocksPerGridL1 * 2) + (NUM_THREADS * 2 - 1)) * sizeof(int);
        if((nvidiaFreeMemory - occupancy) < 0)
            clean = 0;
    } else if(blocksPerGridL1 != 1 && blocksPerGridL3 == 1) {
        double occupancy = ((blocksPerGridL1 + (NUM_THREADS * 2) + (NUM_THREADS * 2 - 1)) * sizeof(int));
        if((nvidiaFreeMemory - occupancy) < 0)
            clean = 0;
    }
    return clean;
}


void prefixSum(int *d_input, int *d_cscColPtr, int numElements) {
	cudaError_t err = cudaSuccess;
    size_t size = numElements * sizeof(int);
	int *d_SUMS_LEVEL1 = NULL;
	int *d_INCR_LEVEL1 = NULL;
	int *d_SUMS_LEVEL2 = NULL;
    int *d_INCR_LEVEL2 = NULL;

	// The correct level is going to be where the SUMS array can be prescanned with only one block
	int blocksPerGridL1 = 1 + (numElements - 1) / (NUM_THREADS * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (NUM_THREADS * 2);
    int blocksPerGridL3 = 1 + blocksPerGridL2 / (NUM_THREADS * 2);
	if(blocksPerGridL1 == 1) {
	    blockPrefixSum<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, NULL);
	} else if (blocksPerGridL2 == 1) {
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");
        
		err = cudaMalloc((void**) &d_INCR_LEVEL1, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");
        
		blockPrefixSum<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, d_SUMS_LEVEL1);
		blockPrefixSum<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, NULL);
		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_cscColPtr, numElements, d_INCR_LEVEL1);
		cudaDeviceSynchronize();
	} else if (blocksPerGridL3 == 1) {
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_SUMS_LEVEL2, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL2");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		err = cudaMalloc((void**) &d_INCR_LEVEL2, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		blockPrefixSum<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, d_SUMS_LEVEL1);
		blockPrefixSum<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, d_SUMS_LEVEL2);
		blockPrefixSum<<<blocksPerGridL3, NUM_THREADS>>>(d_SUMS_LEVEL2, d_INCR_LEVEL2, blocksPerGridL2, NULL);
		uniformAdd<<<blocksPerGridL2, NUM_THREADS>>>(d_INCR_LEVEL1, blocksPerGridL1, d_INCR_LEVEL2);
		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_cscColPtr, numElements, d_INCR_LEVEL1);
		cudaDeviceSynchronize();
	}else {
		printf("The array of length = %d is to large for a level 3 FULL prescan\n", numElements);
    }
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	// Only need to free these arrays if they were allocated
	if(blocksPerGridL2 == 1 || blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL1");
		err = cudaFree(d_INCR_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL1");
	}
	if(blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL2");
		err = cudaFree(d_INCR_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL2");
	}
}


__global__
void blockPrefixSum(int *g_idata, int *g_odata, int n, int *SUM) {
	__shared__ int temp[NUM_THREADS * 2 + (NUM_THREADS)];
	int thid = threadIdx.x;
	int offset = 1;
    int blockOffset = NUM_THREADS * blockIdx.x * 2;
    
	// Create the correct offsets for BCAO
	int ai = thid;
	int bi = thid + NUM_THREADS;

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	// Copy the correct elements form the global array
	if (blockOffset + ai < n) {
        // Load input into shared memory
		temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
	}
	if (blockOffset + bi < n) {
        // Load input into shared memory
		temp[bi + bankOffsetB] = g_idata[blockOffset + bi];
	}

	for (int d = NUM_THREADS; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) {
		if(SUM != NULL) {
			// If doing a FULL scan, save the last value in the SUMS array for later processing
			SUM[blockIdx.x] = temp[(NUM_THREADS * 2) - 1 + CONFLICT_FREE_OFFSET((NUM_THREADS * 2) - 1)];
        }
        // clear the last element
		temp[(NUM_THREADS * 2) - 1 + CONFLICT_FREE_OFFSET((NUM_THREADS * 2) - 1)] = 0;
	}

	// Traverse down tree & build scan
	for (int d = 1; d < NUM_THREADS * 2; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	// Copy the new array back to global array
	__syncthreads();
	if (blockOffset + ai < n) {
        // write results to device memory
		g_odata[blockOffset + ai] = temp[ai + bankOffsetA];
	}
	if (blockOffset + bi < n) {
        // write results to device memory
		g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
	}
}



// Takes the output array and for each block i, adds value i from INCR array to every element
__global__
void uniformAdd(int *outputArray, int numElements, int *INCR) {
	int index = threadIdx.x + (2 * NUM_THREADS) * blockIdx.x;
	int valueToAdd = INCR[blockIdx.x];
	if (index < numElements){
		outputArray[index] += valueToAdd;
	}
	if (index + NUM_THREADS < numElements){
		outputArray[index + NUM_THREADS] += valueToAdd;
	}
}