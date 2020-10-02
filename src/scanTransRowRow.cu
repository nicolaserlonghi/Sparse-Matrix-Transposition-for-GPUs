#include <scanTransRowRow.h>
#include <utilities.h>
#include <Timer.cuh>

using namespace timer;

const int NUM_THREADS = 128;

__global__
void transpostionRowRowKernel(
    int     m,
    int     n,
    int     nnz,
    int     len,
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

        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += inter[(global_id + 1) * n + i];
            inter[(global_id + 1) * n + i] = sum;
        }
    }
}

__global__ 
void transpostionRowRowKernel2(
    int     m,
    int     n,
    int     nnz,
    int     len,
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
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    // in realtà dovrebbe essere il maggiore tra n e m?
    int nthreads = m;
    
    if ( global_id < n) {
        for(int j = 1; j < (nthreads + 1); j++) {
            inter[global_id + (n * j)] += inter[global_id + (n * (j-1))];
        }
        cscColPtr[global_id + 1] = inter[global_id + (n * nthreads)];
    }
}

__global__ 
void transpostionRowRowKernel3(
    int     m,
    int     n,
    int     nnz,
    int     len,
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
            int index = row_offset != 0 ? inter[global_id * n + row_offset] - inter[global_id * n + row_offset - 1] : inter[global_id * n + row_offset];
            loc = cscColPtr[row_offset] + index + intra[start + i];
            cscRowIdx[loc] = csrRowIdx[start + i];
            cscVal[loc] = csrVal[start + i];        
        }        
    }
}

float scanTransRowRow(
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

    double reqMem = (nnz * sizeof(int)) * 4 + (nnz * sizeof(double)) * 2 + (m + 1) * sizeof(int) + (n + 1) * sizeof(int) + ((nthreads + 1) * n) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();    
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
    if ( nvidiaFreeMemory < reqMem) {        
        return -1;
    }

    cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMalloc(&d_csrColIdx, nnz     * sizeof(int));
    cudaMalloc(&d_csrVal,    nnz     * sizeof(double));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz     * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz     * sizeof(double), cudaMemcpyHostToDevice);
   
    cudaMalloc(&d_cscColPtr, (n + 1) * sizeof(int));
    cudaMalloc(&d_cscRowIdx, nnz     * sizeof(int));
    cudaMalloc(&d_cscVal,    nnz     * sizeof(double));  

    cudaMemset(d_cscColPtr, 0, (n + 1) * sizeof(int));
    cudaMemset(d_cscRowIdx, 0, nnz   * sizeof(int));
    cudaMemset(d_cscVal,    0, nnz   * sizeof(double));

    cudaMalloc(&d_csrRowIdx, nnz                * sizeof(int));
    cudaMalloc(&d_intra,     nnz                * sizeof(int));
    cudaMalloc(&d_inter,     (nthreads + 1) * n * sizeof(int));

    cudaMemset(d_csrRowIdx, 0, nnz                  * sizeof(int));
    cudaMemset(d_intra,     0, nnz                  * sizeof(int));
    cudaMemset(d_inter,     0, (nthreads + 1) * n   * sizeof(double));

    TM_device.start();

    dim3 DimGrid(biggest / NUM_THREADS, 1, 1);
    if (biggest % NUM_THREADS) DimGrid.x++;
    dim3 DimBlock(NUM_THREADS, 1, 1);

    int len = nnz / nthreads;

    transpostionRowRowKernel<<<DimGrid, DimBlock>>>(m, n, nnz, len, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    cudaDeviceSynchronize();
    
    transpostionRowRowKernel2<<<DimGrid, DimBlock>>>(m, n, nnz, len, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    cudaDeviceSynchronize();
    
    transpostionRowRowKernel3<<<DimGrid, DimBlock>>>(m, n, nnz, len, d_csrRowPtr, d_csrColIdx, d_csrVal, d_cscColPtr, d_cscRowIdx, d_cscVal, d_csrRowIdx, d_intra, d_inter);

    TM_device.stop();
    cudaDeviceSynchronize();
    TM_device.print("GPU Sparse Matrix Transpostion ScanTrans Row-Row: ");    
  
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_csrRowPtr);
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
