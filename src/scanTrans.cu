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
    double *d_csrVal;
   
    int *d_cscColPtr;
    int *d_cscRowIdx;
    double *d_cscVal;

    int *d_csrRowIdx;    
    int *d_intra;
    int *d_inter;


    // questi ci serono ovunque
    cudaMalloc(&d_csrColIdx, nnz     * sizeof(int));    
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMalloc(&d_cscColPtr, (n + 1) * sizeof(int));    
    
    // questi ci servono nel primo kernel 
    cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMalloc(&d_csrRowIdx, nnz                * sizeof(int));
    cudaMalloc(&d_intra,     nnz                * sizeof(int));
    cudaMalloc(&d_inter,     (biggest + 1) * n * sizeof(int));
    cudaMemset(d_inter,     0, (biggest + 1) * n   * sizeof(int));



    // From here start cuda computation
    Timer<DEVICE> TM_device;
    TM_device.start();
    histogram <<<gridSizeHistogram, blockSizeHistogram>>>(m, n, nnz, d_csrRowPtr, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR

    // Cleaning
    cudaFree(d_csrRowPtr);

    verticalScan <<<gridSizeVerticalScan, blockSizeVerticalScan>>>(m, n, nnz, d_csrColIdx, d_cscColPtr, d_csrRowIdx, d_intra, d_inter);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR

    int clean = manageMemoryForScan(n + 1);
    int *intra;
    int *csrRowIdx;
    if(clean == 0) {
        intra = (int *) malloc(nnz * sizeof(int));
        cudaMemcpy(intra, d_intra, (nnz) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(csrColIdx, d_csrColIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        csrRowIdx  = (int *) malloc(nnz * sizeof(int));
        cudaMemcpy(csrRowIdx, d_csrRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        // Cleaning device
        cudaFree(d_intra);
        cudaFree(d_csrColIdx);
        cudaFree(d_csrRowIdx);
    }
    
    BCAO_fullPrescan(d_inter + (n * biggest), d_cscColPtr, n + 1);
    
    if(clean == 0) {
        cudaMalloc(&d_intra, nnz * sizeof(int));
        cudaMalloc(&d_csrColIdx, nnz * sizeof(int));    
        cudaMalloc(&d_csrRowIdx, nnz * sizeof(int));
        cudaMemcpy(d_intra, intra, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csrRowIdx, csrRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        // Cleaning host
        free(intra);
        free(csrRowIdx);
    }

    // questi ci servono nell'ultimo
    cudaMalloc(&d_csrVal,    nnz     * sizeof(double));
    cudaMemcpy(d_csrVal,    csrVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_cscRowIdx, nnz     * sizeof(int));
    cudaMalloc(&d_cscVal,    nnz     * sizeof(double)); 
    
    writeBack<<<gridSizeWriteBack, blockSizeWriteBack>>>(m, n, nnz, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);
    cudaDeviceSynchronize();

    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ScanTrans: ");    
  
    // Copy results
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleaning 
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);
    cudaFree(d_csrRowIdx);
    cudaFree(d_intra);
    cudaFree(d_inter);

    return TM_device.duration(); 
}

bool checkIsMemoryNotEnough(int n, int m, int nnz) {
    int biggest = m > n ? m : n;
    unsigned long long int reqMem = (nnz * sizeof(int)) * 4 + (nnz * sizeof(double)) * 2 + (biggest + 1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();
    unsigned long long int actualFreeMem = nvidiaFreeMemory - reqMem;
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << " actual: " << actualFreeMem << std::endl;
    unsigned long long int altro = (unsigned long long int)((unsigned long long int)(biggest + 1) * (unsigned long long int)n) * sizeof(int);
    std::cout << std::setprecision(0) << "altro: " << altro << " free memory: " << actualFreeMem << std::endl;
    if ( actualFreeMem < altro) {        
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