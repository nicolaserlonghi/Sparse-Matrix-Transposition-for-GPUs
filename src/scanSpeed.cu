#include <scanSpeed.h>
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
void transpostionSpeedKernel(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    int     *cscColPtr,
    int     *csrRowIdx,
    int     *intra,
    int     *inter,
    int     inter_dim
) {

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int start;
    

    // prendo l'equivalente di biggest della versione seriale
    int  bestNumThread = n > m ? n : m;
    // se biggest è maggiore del numero di thread che ho allocato vuol dire che sarei uscita
    // quindi prendo nthreads (altrimenti detto come "prendo il più piccolo tra bestNumThread e nthreads")
    bestNumThread = inter_dim;
    if (global_id == 0) {
        printf("%d\n", bestNumThread);
    }
    
    int len;

    // if(global_id < bestNumThread) {
    //     len = nnz / bestNumThread;        
    //     // partizioniamo il numero di nnz tra i thread
    //     if ( global_id < nnz % bestNumThread) {                
    //         len++;
    //         start = len * global_id;                 
    //     }
    //     else {
    //         start = len * global_id + (nnz % bestNumThread);    
    //     }
        
    //     if (global_id < m) {
    //         for(int j = csrRowPtr[global_id]; j < csrRowPtr[global_id + 1]; j++) {
    //             csrRowIdx[j] = global_id;
    //         }
    //     }       
        
    //     for(int i = 0; i < len; i++) {
    //         int index = csrColIdx[start + i];
    //         int brutto = (global_id + 1) * n + index;
    //         intra[start + i] = inter[brutto];
    //         inter[brutto]++;
    //     }
    //     // global_id += nthreads;

    

    //     // for(int i = 0; i < len; i++) {
    //     //     int index = csrColIdx[start + i];
    //     //     long long int a = global_id;
    //     //     long long int b = n;
    //     //     long long int c = index;
    //     //     long long int brutto = (a + 1) * b + c;
    //     //     if(brutto > global_id || brutto < 0) {
    //     //         int *ptr = &inter[brutto];
    //     //         // if (x%2048)  ptr = &inter[((global_id + 1) * x + index) + 1];
    //     //         brutto = (global_id + 1) * x + index;
    //     //         intra[start + i] = ptr[brutto];
    //     //         ptr[brutto]++;
    //     //     } else {
    //     //         intra[start + i] = inter[brutto];
    //     //         inter[brutto]++;
    //     //     }
    //     // }
    //     // global_id += nthreads;
    // }




    /* int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int start;
    

    int  bestNumThread = n > m ? n : m;
    int len;

    if (global_id == 0) {
        printf("%ld\n", n * (m - 2));
        printf("%ld\n", n * (m - 1));
        printf("%ld\n", n * (m));
        printf("%d\n", n * (m +1));
    }

    while(global_id < bestNumThread) {
        len = nnz / bestNumThread;
        // printf("best: %d, global_id: %d, len: %d\n", bestNumThread, global_id, len);        
        // partizioniamo il numero di nnz tra i thread
        if ( global_id < nnz % bestNumThread) {                
            len++;
            start = len * global_id;                 
        }
        else {
            start = len * global_id + (nnz % bestNumThread);    
        }
        
        if (global_id < m) {
            for(int j = csrRowPtr[global_id]; j < csrRowPtr[global_id + 1]; j++) {
                csrRowIdx[j] = global_id;
            }
        }        
     

        for(int i = 0; i < len; i++) {
            long index = csrColIdx[start + i];
            // if ( ((global_id + 1) * n + index ) < ((global_id + 1) * n + n) ) {
            
                // printf("%d - %d - %d - %d - %d\n", global_id, len, csrColIdx[start + i], start, (global_id + 1) * n);
                // int index = 0;
                // printf("%d =  %d * %d + %d --> len: %d\n", (global_id + 1) * n + index, global_id, n, index, len);
                long a = (global_id + 1) * n + index;
                // printf("global_id: %d, -> (%d + 1) * %d + %d =  a: %ld --> len: %d, start: %d\n", global_id, global_id, n, index, a, len, start);
                // intra[start + i] = *(inter + a);
                // *(inter + a) += 1;
            // }
        } 
        // printf("pref: %d, after: %d \n", global_id, global_id + nthreads);
        global_id += nthreads;
    } */
}

__global__
void transpostionSpeedKernel1(
    int     m,
    int     n,
    int     nnz,
    int     *csrColIdx,
    int     *cscColPtr,
    int     *csrRowIdx,
    int     *intra,
    int     *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    if (global_id == 0) {
        printf("Ciao");
    }
    
    int  bestNumThread = n > m ? n : m;
    while(global_id < bestNumThread) {
        if (global_id < n) {
            for(int j = 1; j < (bestNumThread + 1); j++) {
                //inter[global_id + (n * j)] += inter[global_id + (n * (j-1))];
            }
        }
        global_id += nthreads;
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




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


void fullPrescan(int *d_input, int *d_cscColPtr, int numElements) {

    
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	size_t size = numElements * sizeof(int);

	// The number of blocks it would take to process the array at each level
	int blocksPerGridL1 = 1 + (numElements - 1) / (NUM_THREADS * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (NUM_THREADS * 2);
    int blocksPerGridL3 = 1 + blocksPerGridL2 / (NUM_THREADS * 2);


	// int *d_input = NULL;
	// err = cudaMalloc((void **) &d_input, size);
    // CUDA_ERROR(err, "Failed to allocate device array x");    

	// int *d_cscColPtr = NULL;
	// err = cudaMalloc((void**) &d_cscColPtr, size);
    // CUDA_ERROR(err, "Failed to allocate device array y");

	// Only define in here and actually allocate memory to these arrays if needed
	int *d_SUMS_LEVEL1 = NULL;
	int *d_INCR_LEVEL1 = NULL;
	int *d_SUMS_LEVEL2 = NULL;
	int *d_INCR_LEVEL2 = NULL;

	// err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    // CUDA_ERROR(err, "Failed to copy array x from host to device");


    //-----------------Pick the correct level and execute the kernels----------
    
    Timer<DEVICE> TM_kernelNew;
    TM_kernelNew.start();

	// The correct level is going to be where the SUMS array can be prescanned with only one block
	if(blocksPerGridL1 == 1){
	    blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, NULL);
        cudaDeviceSynchronize();
	} else if (blocksPerGridL2 == 1) {
        
		// SUMS and INCR arrays need to be allocated to store intermediate values
        err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, blocksPerGridL1 * sizeof(int));
        CUDA_ERROR(err, "Failed to allocate device vector d_INCR_LEVEL1");
        
		blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, d_SUMS_LEVEL1);

		// Run a second prescan on the SUMS array
		blockPrescan<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, NULL);

		// Add the values of INCR array to the corresponding blocks of the d_cscColPtr array
		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_cscColPtr, numElements, d_INCR_LEVEL1);

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

		blockPrescan<<<blocksPerGridL1, NUM_THREADS>>>(d_input, d_cscColPtr, numElements, d_SUMS_LEVEL1);

		blockPrescan<<<blocksPerGridL2, NUM_THREADS>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, d_SUMS_LEVEL2);

		blockPrescan<<<blocksPerGridL3, NUM_THREADS>>>(d_SUMS_LEVEL2, d_INCR_LEVEL2, blocksPerGridL2, NULL);

		uniformAdd<<<blocksPerGridL2, NUM_THREADS>>>(d_INCR_LEVEL1, blocksPerGridL1, d_INCR_LEVEL2);

		uniformAdd<<<blocksPerGridL1, NUM_THREADS>>>(d_cscColPtr, numElements, d_INCR_LEVEL1);

		cudaDeviceSynchronize();
	}else {
		printf("The array of length = %d is to large for a level 3 FULL prescan\n", numElements);
    }

    TM_kernelNew.stop();
    TM_kernelNew.print("Kernel_new: ");
    
    //---------------------------Timing and verification-----------------------

    err = cudaGetLastError();
    CUDA_ERROR(err, "Failed to launch fullPrescan");

    // err = cudaMemcpy(cscColPtr, d_cscColPtr, size, cudaMemcpyDeviceToHost);
    // CUDA_ERROR(err, "Failed to copy array y from device to host");


	//-------------------------------Cleanup-----------------------------------
	// Free device memory
	// err = cudaFree(d_input);
	// CUDA_ERROR(err, "Failed to free device array x");
	// err = cudaFree(d_cscColPtr);
    // CUDA_ERROR(err, "Failed to free device array y");

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
}


int manageMemoryForScan(int numElements){
    // The number of blocks it would take to process the array at each level
	int blocksPerGridL1 = 1 + (numElements - 1) / (NUM_THREADS * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (NUM_THREADS * 2);
    int blocksPerGridL3 = 1 + blocksPerGridL2 / (NUM_THREADS * 2);

    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    int clean = 1;
    if (blocksPerGridL1 != 1 && blocksPerGridL2 == 1) {
        double occupancy = (blocksPerGridL1 * sizeof(int)) * 2;
        if((nvidiaFreeMemory - occupancy) < 0)
            clean = 0;
    } else if(blocksPerGridL1 != 1 && blocksPerGridL3 == 1) {
        double occupancy = ((blocksPerGridL1 + (NUM_THREADS * 2)) * sizeof(int)) * sizeof(int);
        if((nvidiaFreeMemory - occupancy) < 0)
            clean = 0;
    }
    return clean;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








__global__ 
void transpostionSpeedKernel3(
    int     m,
    int     n,
    int     nnz,
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
    int  bestNumThread = n > m ? n : m;
    int len;
  
    while(global_id < bestNumThread) {
        len = nnz / bestNumThread;
        if ( global_id < nnz % bestNumThread) {
            len ++;
            start = len * global_id;    
        }
        else {
            start = len * global_id + (nnz % bestNumThread);
        }

        int loc;
        for(int i = 0; i < len; i++) {
            int row_offset = csrColIdx[start + i];
            int index = inter[global_id * n + row_offset];
            loc = cscColPtr[row_offset] + index + intra[start + i];
            cscRowIdx[loc] = csrRowIdx[start + i];
            cscVal[loc] = csrVal[start + i];        
        }

        global_id += nthreads;
    }
}


float scanSpeed(
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

    int biggest = m > n ? m : n;
    int inter_dim = biggest;

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxThreads = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
    
    if (biggest > maxThreads) {
        inter_dim = 1000;
    }
    std::cout << biggest << std::endl;
    

    // int nthreads = biggest / NUM_THREADS;
    // if(biggest % NUM_THREADS) nthreads++;
    // nthreads = nthreads * NUM_THREADS;
    // nthreads = biggest;
    
    int     *d_csrRowPtr;
    int     *d_csrColIdx;
    double  *d_csrVal;
   
    int     *d_cscColPtr;
    int     *d_cscRowIdx;
    double  *d_cscVal;

    int     *d_csrRowIdx;    
    int     *d_intra;
    int     *d_inter;

    // int device;
    // cudaGetDevice(&device);

    // struct cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, device);

    // int maxThreads = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
    // int maxBlocks = props.multiProcessorCount * 16;    

    // int requiredBlocks = biggest / NUM_THREADS;
    // if(biggest % NUM_THREADS) requiredBlocks++;
    // int requiredThreads = requiredBlocks * NUM_THREADS;

    // int blockNum;

    // if (requiredThreads > maxThreads) {
    //     std::cout << "numero di thread richiesto eccessivo: " << requiredThreads << " su " << maxThreads << " " << std::endl;
    //     std::cout << "eseguiamo comunque il tutto limitando il numero di thread al massimo possibile ed eseguendo più chiamate ai kernel" << std::endl;
    //     std::cout << "questo potrebbe avere impatto sulle performance" << std::endl;
    //     blockNum = maxThreads / NUM_THREADS;
    //     std::cout << NUM_THREADS << " " << blockNum << std::endl;
    // }
    // else {
    //     blockNum = requiredBlocks;
    //     std::cout << blockNum << " " << biggest <<std::endl;
    // }

    unsigned long long int reqMem = (nnz * sizeof(int)) * 4 + (nnz * sizeof(double)) * 2 + (biggest + 1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    unsigned long long int actualFreeMem = nvidiaFreeMemory - reqMem;
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << " actual: " << actualFreeMem << std::endl;
    unsigned long long int altro = (unsigned long long int)((unsigned long long int)(biggest + 1) * (unsigned long long int)n) * sizeof(int);    
    // if ( (actualFreeMem / n) < biggest) {
    //     std::cout << "Questa matrice non ci sta in memoria con un numero di righe pari a " << biggest << " per cui la allocheremo più piccola" << std::endl;
    //     inter_dim = (actualFreeMem / n) - 1;
    //     std::cout << inter_dim << std::endl;
    // }
    std::cout << std::setprecision(0) << "altro: " << altro << " free memory: " << actualFreeMem << std::endl;
    if ( actualFreeMem < altro) {        
        return -1;
    }

    // questi ci serono ovunque
    cudaMalloc(&d_csrColIdx, nnz     * sizeof(int));    
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMalloc(&d_cscColPtr, (n + 1) * sizeof(int));    
    
    // questi ci servono nel primo kernel 
    cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrRowIdx, nnz                * sizeof(int));
    cudaMalloc(&d_intra,     nnz                * sizeof(int));
    cudaMalloc(&d_inter,     (inter_dim + 1) * n * sizeof(int));
    cudaMemset(d_inter,     0, (inter_dim + 1) * n   * sizeof(int));

    int blockSize1;
    int minGridSize1;
    int gridSize1;
    int blockSize2;
    int minGridSize2;
    int gridSize2;
    int blockSize3;
    int minGridSize3;
    int gridSize3;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, transpostionSpeedKernel, 0, biggest);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, transpostionSpeedKernel1, 0, biggest);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize3, &blockSize3, transpostionSpeedKernel3, 0, biggest);

    gridSize1 = (biggest + blockSize1 - 1) / blockSize1;
    gridSize2 = (biggest + blockSize2 - 1) / blockSize2;
    gridSize3 = (biggest + blockSize3 - 1) / blockSize3;


    std::cout << "blockSize1: " << blockSize1 << " minGridSize1: " << minGridSize1  << " gridSize1: " << gridSize1 << std::endl;
    std::cout << "blockSize2: " << blockSize2 << " minGridSize2: " << minGridSize2  << " gridSize2: " << gridSize2 << std::endl;
    std::cout << "blockSize3: " << blockSize3 << " minGridSize3: " << minGridSize3  << " gridSize3: " << gridSize3 << std::endl;

    TM_device.start();

    // dim3 DimGrid(blockNum, 1, 1);
    // dim3 DimBlock(NUM_THREADS, 1, 1);

    transpostionSpeedKernel<<<gridSize1, blockSize1>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter, inter_dim);

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR

    // non ci servirà più
    cudaFree(d_csrRowPtr);

    transpostionSpeedKernel1<<<gridSize2, blockSize2>>>(m, n, nnz, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter);

    cudaDeviceSynchronize();

    std::cout << "### Before: " << getSizeOfNvidiaFreeMemory() << std::endl;
    
    int clean = manageMemoryForScan(n + 1);
    int *intra;
    int *csrRowIdx;
    if(clean == 0) {
        intra  = (int *)malloc(nnz * sizeof(int));
        cudaMemcpy(intra, d_intra, (nnz) * sizeof(int),  cudaMemcpyDeviceToHost);
        cudaFree(d_intra);
        
        cudaMemcpy(csrColIdx, d_csrColIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
        cudaFree(d_csrColIdx);
        
        csrRowIdx  = (int *)malloc(nnz * sizeof(int));
        cudaMemcpy(csrRowIdx, d_csrRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
        cudaFree(d_csrRowIdx);
    }
    
    std::cout << "### After: " << getSizeOfNvidiaFreeMemory() << std::endl;
    
    fullPrescan(d_inter + (n * biggest), d_cscColPtr, n + 1);
    
    if(clean == 0) {
        cudaMalloc(&d_intra,     nnz                * sizeof(int));
        cudaMemcpy(d_intra, intra, nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    
        cudaMalloc(&d_csrColIdx, nnz     * sizeof(int));    
        cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    
        cudaMalloc(&d_csrRowIdx, nnz                * sizeof(int));
        cudaMemcpy(d_csrRowIdx, csrRowIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice);

        free(intra);
        free(csrRowIdx);
    }

    // questi ci servono nell'ultimo
    cudaMalloc(&d_csrVal,    nnz     * sizeof(double));
    cudaMemcpy(d_csrVal,    csrVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_cscRowIdx, nnz     * sizeof(int));
    cudaMalloc(&d_cscVal,    nnz     * sizeof(double)); 

    cudaDeviceSynchronize();   
    
    transpostionSpeedKernel3<<<gridSize3, blockSize3>>>(m, n, nnz, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    TM_device.stop();
    cudaDeviceSynchronize();
    TM_device.print("GPU Sparse Matrix Transpostion ScanSpeed: ");    
  
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);

    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);

    cudaFree(d_csrRowIdx);
    cudaFree(d_intra);
    cudaFree(d_inter);


      ////////////////////////////////////// For debug ///////////////////////////////////////

    //   //int *intra  = (int *)malloc(nnz * sizeof(int));
    //   int *inter  = (int *)malloc((biggest + 1) * n * sizeof(int) * sizeof(int));
    //   //int *csrRowIdx  = (int *)malloc(nnz * sizeof(int));
    //   cudaMemcpy(intra, d_intra, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    //   cudaMemcpy(inter, d_inter, (biggest + 1) * n * sizeof(int), cudaMemcpyDeviceToHost);
    //   cudaMemcpy(csrRowIdx, d_csrRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
  
    //   std::cout << "intra: ";
    //   for(int i = 0; i < nnz; i++) {
    //       std::cout << intra[i] << " ";
    //   }
    //   std::cout << std::endl;
  
    //   std::cout << "inter: ";
    //   for(int i = 0; i < ((biggest + 1) * n); i++) {
    //       std::cout << inter[i] << " ";
    //   }
    //   std::cout << std::endl;
      
    //   std::cout << "csrRowIdx: ";
    //   for(int i = 0; i < nnz; i++) {
    //       std::cout << csrRowIdx[i] << " ";
    //   }
    //   std::cout << std::endl;
  
    //   //free(intra);
    //   free(inter);
    //   //free(csrRowIdx);
  
      ////////////////////////////////////////////////////////////////////////////////////////




    return TM_device.duration(); 
}
