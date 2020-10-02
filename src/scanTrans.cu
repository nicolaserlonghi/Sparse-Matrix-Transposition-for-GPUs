#include <scanTrans.h>
#include <utilities.h>
#include <Timer.cuh>
#include "CheckError.cuh"

using namespace timer;

const int NUM_THREADS = 128;

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
    if (err != cudaSuccess) {\
        printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
        exit( EXIT_FAILURE );\
    }\
}

__global__
void transpostionKernel(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal,
    int     *csrRowIdx,
    int     *intra,
    int     *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    // in realtà dovrebbe essere il maggiore tra n e m?
    nthreads = m;
    int start;
    int len = nnz / nthreads;

    if ( global_id < nnz % nthreads) {
        len ++;
        start = len * global_id;    
    }
    else {
        start = len * global_id + (nnz % nthreads);    
    }

    if (global_id < nthreads) {
        for(int j = csrRowPtr[global_id]; j < csrRowPtr[global_id + 1]; j++) {
            csrRowIdx[j] = global_id;
        }

        for(int i = 0; i < len; i++) {
            int index = csrColIdx[start + i];
            intra[start + i] = inter[(global_id + 1) * n + index];
            inter[(global_id + 1) * n + index]++;
        } 
    }
}

__global__
void transpostionKernel1(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal,
    int     *csrRowIdx,
    int     *intra,
    int     *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    // in realtà dovrebbe essere il maggiore tra n e m?
    nthreads = m;

    if (global_id < n) {
        for(int j = 1; j < (nthreads + 1); j++) {
            inter[global_id + (n * j)] += inter[global_id + (n * (j-1))];
        }
    }
}

//// PREFIX SUM ////
__global__ 
void transpostionKernel2(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal,
    int     *csrRowIdx,
    int     *intra,
    int     *inter
) {

    __shared__ int temp[800]; // allocated on invocation
	int thid = threadIdx.x + NUM_THREADS * blockIdx.x * 2;
	int offset = 1;

//	 Copy the correct elements form the global array
	if ((thid * 2) < n+1){
        temp[thid * 2] = inter[n*m + (thid * 2)];
	}
	if ((thid * 2) + 1 < n+1){
        temp[(thid * 2) + 1] = inter[n*m + (thid * 2) + 1];
	}

//	 Build sum in place up the tree
	for (int d = NUM_THREADS; d > 0; d >>= 1){
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (thid == 0) {
		temp[(NUM_THREADS << 1) - 1] = 0; // clear the last element
	}

//	 Traverse down tree & build scan
	for (int d = 1; d < NUM_THREADS << 1; d <<= 1){
		offset >>= 1;
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
    }


//	 Copy the new array back to global array
    __syncthreads();
    // printf("?? thid: %d \n", thid * 2 );
	if ((thid * 2) < (n + 1)){
        printf("## thid: %d \n", thid * 2 );
        cscColPtr[thid * 2] = temp[(thid * 2)]; // write results to device memory
	}
	if ((thid * 2) + 1 < n + 1){
        cscColPtr[((thid * 2)+1)] = temp[(thid * 2)+1];
        // printf("## %d \n", thid * 2+1);
	}





    /*
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    // in realtà dovrebbe essere il maggiore tra n e m?
    int nthreads = m;
    
	if (global_id > n) {
		return;
	}
	int sum = 0;
	for (int i = 0; i < global_id + 1; ++i) {
		sum = sum + inter[nthreads * n + i];
	}
    cscColPtr[global_id + 1] = sum;
    if(global_id + 1 == 10 || global_id + 1 == 20 || global_id + 1 == 100)
        printf("old version: cscColPtr[%d]: %d \n", global_id + 1, cscColPtr[global_id + 1]);

    */
}

__global__ 
void transpostionKernel3(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal,
    int     *csrRowIdx,
    int     *intra,
    int     *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;    
    int nthreads = blockDim.x * gridDim.x;
    int start;
    // in realtà dovrebbe essere il maggiore tra n e m?
    nthreads = m;
    int len = nnz / nthreads;

    if(global_id < n +1) {
        if(global_id == 10 || global_id == 20 || global_id == 100)
            printf("old version: cscColPtr[%d]: %d \n", global_id , cscColPtr[global_id ]);
    }

    if ( global_id < nnz % nthreads) {
        len ++;
        start = len * global_id;    
    }
    else {
        start = len * global_id + (nnz % nthreads);
    }

    int loc;
    if ( global_id < nthreads) { 
        for(int i = 0; i < len; i++) {
            int row_offset = csrColIdx[start + i];
            int index = inter[global_id * n + row_offset];
            loc = cscColPtr[row_offset] + index + intra[start + i];
            cscRowIdx[loc] = csrRowIdx[start + i];
            cscVal[loc] = csrVal[start + i];        
        }        
    }
}














// Takes the output array and for each block i, adds value i from INCR array to every element
__global__
void uniformAdd(int *outputArray, int numElements, int *INCR){
	int index = threadIdx.x + (2 * NUM_THREADS) * blockIdx.x;

	int valueToAdd = INCR[blockIdx.x];

	// Each thread sums two elements
	if (index < numElements){
		outputArray[index] += valueToAdd;
	}
	if (index + NUM_THREADS < numElements){
		outputArray[index + NUM_THREADS] += valueToAdd;
	}
}

// Block prescan that works on any array length on NUM_THREADS * 2 length blocks
__global__
void blockPrescan(int *g_idata, int *g_odata, int n, int *SUM)
{
	__shared__ int temp[NUM_THREADS << 1]; // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	int blockOffset = NUM_THREADS * blockIdx.x * 2;

//	 Copy the correct elements form the global array
	if (blockOffset + (thid * 2) < n){
        temp[thid * 2] = g_idata[blockOffset + (thid * 2)];
	}
	if (blockOffset + (thid * 2) + 1 < n){
        temp[(thid * 2)+1] = g_idata[blockOffset + (thid * 2)+1];
	}

//	 Build sum in place up the tree
	for (int d = NUM_THREADS; d > 0; d >>= 1){
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (thid == 0) {
		if(SUM != NULL){
			// If doing a FULL scan, save the last value in the SUMS array for later processing
			SUM[blockIdx.x] = temp[(NUM_THREADS << 1) - 1];
		}
		temp[(NUM_THREADS << 1) - 1] = 0; // clear the last element
	}

//	 Traverse down tree & build scan
	for (int d = 1; d < NUM_THREADS << 1; d <<= 1){
		offset >>= 1;
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

//	 Copy the new array back to global array
	__syncthreads();
	if (blockOffset + (thid * 2) < n){
        g_odata[blockOffset + (thid * 2)] = temp[(thid * 2)]; // write results to device memory
	}
	if (blockOffset + (thid * 2) + 1 < n){
        g_odata[blockOffset + ((thid * 2)+1)] = temp[(thid * 2)+1];
	}
}

__host__
void fullPrescan(int *h_x, int *h_y, int numElements) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	size_t size = numElements * sizeof(int);

	// The number of blocks it would take to process the array at each level
	int blocksPerGridL1 = 1 + (numElements - 1) / (NUM_THREADS * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (NUM_THREADS * 2);
    int blocksPerGridL3 = 1 + blocksPerGridL2 / (NUM_THREADS * 2);

	int *d_x = NULL;
	err = cudaMalloc((void **) &d_x, size);
	CUDA_ERROR(err, "Failed to allocate device array x");

	int *d_y = NULL;
	err = cudaMalloc((void**) &d_y, size);
	CUDA_ERROR(err, "Failed to allocate device array y");

	// Only define in here and actually allocate memory to these arrays if needed
	int *d_SUMS_LEVEL1 = NULL;
	int *d_INCR_LEVEL1 = NULL;
	int *d_SUMS_LEVEL2 = NULL;
	int *d_INCR_LEVEL2 = NULL;

	err = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array x from host to device");


	//-----------------Pick the correct level and execute the kernels----------

	// The correct level is going to be where the SUMS array can be prescanned with only one block
	if(blocksPerGridL1 == 1){
	    blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_x, d_y, numElements, NULL);
        cudaDeviceSynchronize();
	} else if (blocksPerGridL2 == 1) {
        
		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to free device array x");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, blocksPerGridL1 * sizeof(int));
        CUDA_ERROR(err, "Failed to allocate device vector d_INCR_LEVEL1");
        
		blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		// Run a second prescan on the SUMS array
		blockPrescan<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, NULL);

		// Add the values of INCR array to the corresponding blocks of the d_y array
		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_y, numElements, d_INCR_LEVEL1);

		cudaDeviceSynchronize();

	} else if (blocksPerGridL3 == 1) {
		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_SUMS_LEVEL2, (NUM_THREADS * 2) * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL2");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		err = cudaMalloc((void**) &d_INCR_LEVEL2, (NUM_THREADS * 2)* sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		blockPrescan<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, d_SUMS_LEVEL2);

		blockPrescan<<<blocksPerGridL3, NUM_THREADS>>>(d_SUMS_LEVEL2, d_INCR_LEVEL2, blocksPerGridL2, NULL);

		uniformAdd<<<blocksPerGridL2, NUM_THREADS>>>(d_INCR_LEVEL1, blocksPerGridL1, d_INCR_LEVEL2);

		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_y, numElements, d_INCR_LEVEL1);

		cudaDeviceSynchronize();
	}else {
		printf("The array of length = %d is to large for a level 3 FULL prescan\n", numElements);
    }
    
    //---------------------------Timing and verification-----------------------

    err = cudaGetLastError();
    CUDA_ERROR(err, "Failed to launch fullPrescan");

    err = cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array y from device to host");


	//-------------------------------Cleanup-----------------------------------
	// Free device memory
	err = cudaFree(d_x);
	CUDA_ERROR(err, "Failed to free device array x");
	err = cudaFree(d_y);
	CUDA_ERROR(err, "Failed to free device array y");

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

	// Reset the device
	//err = cudaDeviceReset();
    //CUDA_ERROR(err, "Failed to reset the device");

    free(d_x);
    free(d_y);
    free(d_SUMS_LEVEL1);
    free(d_INCR_LEVEL1);
    free(d_SUMS_LEVEL2);
    free(d_INCR_LEVEL2);
}


float scanTrans(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal
) {
    Timer<DEVICE> TM_device;

    // QUESTO CALCOLO È QUI PER L'ALLOCAZIONE DI INTER
    // NEL CASO DECIDESSIMO CHE IL NUMERO DI THREAD È SEMPRE UGUALE A M POSSIAMO TOGLIERLO E METTERE OVUNQUE M
    // Io voglio m thread
    // m / la dimensione di un blocco mi da il numero di blocchi
    int nthreads = m / NUM_THREADS;
    if(m % NUM_THREADS) nthreads++;
    // questo mi dice il numero totale di thread create
    nthreads = nthreads * NUM_THREADS;
    // ma io ne voglio solo m
    // in realtà dovrebbe essere il maggiore tra n e m?
    nthreads = m;

    int biggest = m > n ? m : n;
    
    int     *d_csrRowPtr;
    int     *d_csrColIdx;
    double  *d_csrVal;
   
    int     *d_cscColPtr;
    int     *d_cscRowIdx;
    double  *d_cscVal;

    int     *d_csrRowIdx;    
    int     *d_intra;
    int     *d_inter;

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxThreads = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
    int maxBlocks = props.multiProcessorCount * 16;

    int requiredBlocks = biggest / NUM_THREADS;
    if(biggest % NUM_THREADS) requiredBlocks++;
    int requiredThreads = requiredBlocks * NUM_THREADS;

    // usciamo se chiediamo troppi blocchi
    if (requiredBlocks > maxBlocks) {
        std::cout << "numero di blocchi richiesto eccessivo: " << requiredBlocks << " su " << maxBlocks << std::endl;
        return -2;
    }

    // usciamo se chiediamo troppi thread
    // questo controllo potrebbe non servire se eseguiamo la spartizione di n, m, nnz
    if (requiredThreads > maxThreads) {
        std::cout << "numero di thread richiesto eccessivo: " << requiredThreads << " su " << maxThreads << std::endl;
        return -3;
    }


    double reqMem = (nnz * sizeof(int)) * 4 + (nnz * sizeof(double)) * 2 + (m + 1) * sizeof(int) + (n + 1) * sizeof(int) + ((nthreads + 1) * n) * sizeof(int);    
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
    if ( nvidiaFreeMemory < reqMem) {        
        return -1;
    }

    SAFE_CALL(cudaMalloc(&d_csrColIdx, nnz     * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrVal,    nnz     * sizeof(double)));

    SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrVal,    csrVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice));
   
    SAFE_CALL(cudaMalloc(&d_cscColPtr, (n + 1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscRowIdx, nnz     * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscVal,    nnz     * sizeof(double))); 

    SAFE_CALL(cudaMemset(d_cscColPtr, 0, (n + 1) * sizeof(int)));
    SAFE_CALL(cudaMemset(d_cscRowIdx, 0, nnz   * sizeof(int)));
    SAFE_CALL(cudaMemset(d_cscVal,    0, nnz   * sizeof(double)));

    SAFE_CALL(cudaMalloc(&d_csrRowIdx, nnz                * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_intra,     nnz                * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_inter,     (nthreads + 1) * n * sizeof(int)));

    SAFE_CALL(cudaMemset(d_csrRowIdx, 0, nnz                  * sizeof(int)));
    SAFE_CALL(cudaMemset(d_intra,     0, nnz                  * sizeof(int)));
    SAFE_CALL(cudaMemset(d_inter,     0, (nthreads + 1) * n   * sizeof(int)));

    TM_device.start();

    dim3 DimGrid(biggest / NUM_THREADS, 1, 1);
    if (biggest % NUM_THREADS) DimGrid.x++;
    dim3 DimBlock(NUM_THREADS, 1, 1);

    transpostionKernel<<<DimGrid, DimBlock>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    
    CHECK_CUDA_ERROR

    transpostionKernel1<<<DimGrid, DimBlock>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);


    CHECK_CUDA_ERROR
    // cudaDeviceSynchronize();


    //transpostionKernel2<<<DimGrid, DimBlock>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);


    // int *csrRowIdx = (int *)malloc(nnz * sizeof(int));

    // SAFE_CALL(cudaMemcpy(csrRowIdx, d_csrRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));

    int *inter = (int *)malloc((nthreads + 1) * n * sizeof(int));
    // int *intra = (int *)malloc(nnz * sizeof(int));

    // SAFE_CALL(cudaMemcpy(intra, d_intra, nnz     * sizeof(int),    cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(inter, d_inter, (nthreads + 1) * n * sizeof(int), cudaMemcpyDeviceToHost));


    int *input = (int *)malloc((n + 1) * sizeof(int));
    for(int i = n * m; i < n * m + n; i++) {
        input[i - (n*m)] = inter[i];
    }

    //fullPrescan(input,cscColPtr, n+1);

    
    CHECK_CUDA_ERROR


    // SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_csrVal,    csrVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice));

    // SAFE_CALL(cudaMemcpy(d_cscColPtr, cscColPtr, (n + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_cscRowIdx, cscRowIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_cscVal,    cscVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice));

    // SAFE_CALL(cudaMemcpy(d_csrRowIdx, cscRowIdx, nnz * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_intra, intra, nnz     * sizeof(int),    cudaMemcpyHostToDevice));
    // SAFE_CALL(cudaMemcpy(d_inter, inter, (nthreads + 1) * n * sizeof(int), cudaMemcpyHostToDevice));

    
    printArray(n + 1, cscColPtr);


    // ​cudaError_t err = cudaGetLastError();
    // std::cout << err << std::endl;
    // std::cout << cudaGetErrorString(err) << std::endl;
    
    transpostionKernel3<<<DimGrid, DimBlock>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("GPU Sparse Matrix Transpostion ScanTrans: ");    
  
    SAFE_CALL(cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost));

    SAFE_CALL(cudaFree(d_csrRowPtr));
    SAFE_CALL(cudaFree(d_csrColIdx));
    SAFE_CALL(cudaFree(d_csrVal));

    SAFE_CALL(cudaFree(d_cscColPtr));
    SAFE_CALL(cudaFree(d_cscRowIdx));
    SAFE_CALL(cudaFree(d_cscVal));

    SAFE_CALL(cudaFree(d_csrRowIdx));
    SAFE_CALL(cudaFree(d_intra));
    SAFE_CALL(cudaFree(d_inter));

    free(inter);
    free(input);
    // free(intra);
    // free(csrRowIdx);

    return TM_device.duration(); 
}
