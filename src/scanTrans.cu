#include <scanTrans.h>
#include <utilities.h>
#include <Timer.cuh>
#include "CheckError.cuh"
#include <prefixSum.cuh>

using namespace timer;

///////////////////////////// Function prototypes //////////////////////////////
__global__
void histogram(
    int m,
    int n,
    int nnz,
    int *csrRowPtr,
    int *csrColIdx,
    int *cscColPtr,
    int *csrRowIdx,
    int *intra,
    int *inter
);
__global__
void verticalScan(
    int m,
    int n,
    int nnz,
    int *csrColIdx,
    int *cscColPtr,
    int *csrRowIdx,
    int *intra,
    int *inter
);
__global__ 
void writeBack(
    int m,
    int n,
    int nnz,
    int *csrColIdx,
    double *csrVal,
    int *cscColPtr,
    int *cscRowIdx,
    double *cscVal,
    int *csrRowIdx,
    int *intra,
    int *inter
);

////////////////////////////////////////////////////////////////////////////////


float scanTrans(
    int m,
    int n,
    int nnz,
    int *csrRowPtr,
    int *csrColIdx,
    double *csrVal,
    int *cscColPtr,
    int *cscRowIdx,
    double *cscVal
) {
    int biggest = m > n ? m : n;
    bool isMemoryNotEnough = checkIsMemoryNotEnough(n, m, nnz);
    if(isMemoryNotEnough)
        return -1;
    int gridSizeHistogram;
    int gridSizeVerticalScan;
    int gridSizeWriteBack;
    int blockSizeHistogram;
    int blockSizeVerticalScan;
    int blockSizeWriteBack;
    getOccupancyMaxPotentialBlockSize(
        biggest, 
        &gridSizeHistogram, 
        &gridSizeVerticalScan,
        &gridSizeWriteBack,
        &blockSizeHistogram,
        &blockSizeVerticalScan,
        &blockSizeWriteBack
    );

    int *d_csrRowPtr;
    int *d_csrColIdx;
    int *d_cscColPtr;
    int *d_csrRowIdx;    
    int *d_intra;
    int *d_inter;

    // Set memory to device
    SAFE_CALL( cudaMalloc(&d_csrColIdx, nnz * sizeof(int)) ); 
    SAFE_CALL( cudaMalloc(&d_cscColPtr, (n + 1) * sizeof(int)) ); 
    SAFE_CALL( cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&d_csrRowIdx, nnz * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&d_intra, nnz * sizeof(int)) );
    SAFE_CALL( cudaMalloc(&d_inter, (biggest + 1) * n * sizeof(int)) );
    SAFE_CALL( cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemset(d_inter, 0, (biggest + 1) * n * sizeof(int)) );

///////////////////////// From here start computation  /////////////////////////

    Timer<DEVICE> TM_device;
    TM_device.start();
    histogram <<<gridSizeHistogram, blockSizeHistogram>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter);
    CHECK_CUDA_ERROR
    // Cleaning
    SAFE_CALL( cudaFree(d_csrRowPtr) );

    verticalScan <<<gridSizeVerticalScan, blockSizeVerticalScan>>>(m, n, nnz, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter);
    CHECK_CUDA_ERROR

    // Manage memory for prefixSum
    int haveToclean = manageMemoryForPrefixSum(n + 1);
    int *intra;
    int *csrRowIdx;
    if(haveToclean == 0) {
        intra = (int *) malloc(nnz * sizeof(int));
        SAFE_CALL( cudaMemcpy(intra, d_intra, (nnz) * sizeof(int), cudaMemcpyDeviceToHost) );
        SAFE_CALL( cudaMemcpy(csrColIdx, d_csrColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
        csrRowIdx  = (int *) malloc(nnz * sizeof(int));
        SAFE_CALL( cudaMemcpy(csrRowIdx, d_csrRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
        // Cleaning device
        SAFE_CALL( cudaFree(d_intra) );
        SAFE_CALL( cudaFree(d_csrColIdx) );
        SAFE_CALL( cudaFree(d_csrRowIdx) );
    }
    
    prefixSum(d_inter + (n * biggest), d_cscColPtr, n + 1);
    
    // Cleaning memory after prefixSum
    if(haveToclean == 0) {
        SAFE_CALL( cudaMalloc(&d_intra, nnz * sizeof(int)) );
        SAFE_CALL( cudaMalloc(&d_csrColIdx, nnz * sizeof(int)) );
        SAFE_CALL( cudaMalloc(&d_csrRowIdx, nnz * sizeof(int)) );
        SAFE_CALL( cudaMemcpy(d_intra, intra, nnz * sizeof(int), cudaMemcpyHostToDevice) );
        SAFE_CALL( cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice) );
        SAFE_CALL( cudaMemcpy(d_csrRowIdx, csrRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice) );
        // Cleaning host
        free(intra);
        free(csrRowIdx);
    }

    // Set device for writeBack computation
    double *d_csrVal;
    double *d_cscVal;
    int *d_cscRowIdx;
    SAFE_CALL( cudaMalloc(&d_csrVal, nnz * sizeof(double)) );
    SAFE_CALL( cudaMalloc(&d_cscVal, nnz * sizeof(double)) ); 
    SAFE_CALL( cudaMalloc(&d_cscRowIdx, nnz * sizeof(int)) );
    SAFE_CALL( cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    
    writeBack<<<gridSizeWriteBack, blockSizeWriteBack>>>(m, n, nnz, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);
    CHECK_CUDA_ERROR

    // Take time
    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ScanTrans: ");    
    // Copy results
    SAFE_CALL( cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int), cudaMemcpyDeviceToHost) );
    SAFE_CALL( cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    SAFE_CALL( cudaMemcpy(cscVal, d_cscVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    // Cleaning 
    SAFE_CALL( cudaFree(d_csrColIdx) );
    SAFE_CALL( cudaFree(d_csrVal) );
    SAFE_CALL( cudaFree(d_cscColPtr) );
    SAFE_CALL( cudaFree(d_cscRowIdx) );
    SAFE_CALL( cudaFree(d_cscVal) );
    SAFE_CALL( cudaFree(d_csrRowIdx) );
    SAFE_CALL( cudaFree(d_intra) );
    SAFE_CALL( cudaFree(d_inter) );

    return TM_device.duration(); 
}

bool checkIsMemoryNotEnough(int n, int m, int nnz) {
    int biggest = m > n ? m : n;
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    unsigned long long int reqMem = 
        (nnz * sizeof(int)) * 4 + (nnz * sizeof(double)) * 2 + (biggest + 1) * sizeof(int);
    unsigned long long int actualFreeMem = nvidiaFreeMemory - reqMem;
    unsigned long long int interDimension = (unsigned long long int)(
        (unsigned long long int)(biggest + 1) * (unsigned long long int)n) * sizeof(int);
    if (actualFreeMem < interDimension) {        
        return true;
    }
    return false;
}

void getOccupancyMaxPotentialBlockSize(
    int biggest,
    int *gridSizeHistogram,
    int *gridSizeVerticalScan,
    int *gridSizeWriteBack,
    int *blockSizeHistogram,
    int *blockSizeVerticalScan,
    int *blockSizeWriteBack
) {
    int minGridSizeHistogram;
    int minGridSizeVerticalScan;
    int minGridSizeWriteBack;
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeHistogram, blockSizeHistogram, histogram, 0, biggest);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeVerticalScan, blockSizeVerticalScan, verticalScan, 0, biggest);
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeWriteBack, blockSizeWriteBack, writeBack, 0, biggest);
    *gridSizeHistogram = (biggest + *blockSizeHistogram - 1) / *blockSizeHistogram;
    *gridSizeVerticalScan = (biggest + *blockSizeVerticalScan - 1) / *blockSizeVerticalScan;
    *gridSizeWriteBack = (biggest + *blockSizeWriteBack - 1) / *blockSizeWriteBack;
}

__global__
void histogram(
    int m,
    int n,
    int nnz,
    int *csrRowPtr,
    int *csrColIdx,
    int *cscColPtr,
    int *csrRowIdx,
    int *intra,
    int *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int bestNumThread = n > m ? n : m;
    int start;
    int len;
    int index;

    while(global_id < bestNumThread) {
        len = nnz / bestNumThread;
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
            index = csrColIdx[start + i];
            intra[start + i] = inter[(global_id + 1) * n + index];
            inter[(global_id + 1) * n + index]++;
        } 
        global_id += nthreads;
    }
}

__global__
void verticalScan(
    int m,
    int n,
    int nnz,
    int *csrColIdx,
    int *cscColPtr,
    int *csrRowIdx,
    int *intra,
    int *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;    
    int bestNumThread = n > m ? n : m;
    while(global_id < bestNumThread) {
        if (global_id < n) {
            for(int j = 1; j < (bestNumThread + 1); j++) {
                inter[global_id + (n * j)] += inter[global_id + (n * (j-1))];
            }
        }
        global_id += nthreads;
    }
}

__global__ 
void writeBack(
    int m,
    int n,
    int nnz,
    int *csrColIdx,
    double *csrVal,
    int *cscColPtr,
    int *cscRowIdx,
    double *cscVal,
    int *csrRowIdx,
    int *intra,
    int *inter
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;    
    int nthreads = blockDim.x * gridDim.x;
    int bestNumThread = n > m ? n : m;
    int start;
    int len;
    int loc;
  
    while(global_id < bestNumThread) {
        len = nnz / bestNumThread;
        if ( global_id < nnz % bestNumThread) {
            len ++;
            start = len * global_id;    
        }
        else {
            start = len * global_id + (nnz % bestNumThread);
        }

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