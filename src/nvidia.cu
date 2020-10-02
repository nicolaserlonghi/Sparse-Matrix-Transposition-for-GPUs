#include <nvidia.h>
#include <utilities.h>
#include <Timer.cuh>

using namespace timer;

float nvidia(
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
    cudaError_t err = cudaSuccess;
    cudaSetDevice(0);
    // Check if memory is enough
    double reqMem = (nnz * sizeof(int)) * 2 + (nnz * sizeof(double)) * 2 + (m+1) * sizeof(int) + (n+1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();    
    if ( nvidiaFreeMemory < reqMem)
        return -1;
        
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    int     *d_csrRowPtr;
    int     *d_csrColIdx;
    double  *d_csrVal;
    int     *d_cscColPtr;
    int     *d_cscRowIdx;
    double  *d_cscVal;
    // Set device memory
    err = cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscColPtr");
    err = cudaMalloc(&d_cscRowIdx, nnz * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscRowIdx");
    err = cudaMalloc(&d_cscVal, nnz * sizeof(double));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscVal");
    err = cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrRowPtr");
    err = cudaMalloc(&d_csrColIdx, nnz * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrColIdx");
    err = cudaMalloc(&d_csrVal, nnz * sizeof(double));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrVal");
    err = cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrRowPtr from host to device");
    err = cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrColIdx from host to device");
    err = cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrVal from host to device");
    
    // setup buffersize
    size_t  P_bufferSize = 0;
    char*   p_buffer= NULL;

    cusparseCsr2cscEx2_bufferSize(
        handle,
        m,
        n,
        nnz,
        d_csrVal,
        d_csrRowPtr,
        d_csrColIdx,
        d_cscVal,
        d_cscColPtr,
        d_cscRowIdx,
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &P_bufferSize
    );

    reqMem = reqMem + static_cast<double>(P_bufferSize);
    if (nvidiaFreeMemory < reqMem) {
        err = cudaFree(d_csrRowPtr);
        CUDA_ERROR(err, "Failed to free device array d_csrRowPtr");
        err = cudaFree(d_csrColIdx);
        CUDA_ERROR(err, "Failed to free device array d_csrColIdx");
        err = cudaFree(d_csrVal);
        CUDA_ERROR(err, "Failed to free device array d_csrVal");
        err = cudaFree(d_cscColPtr);
        CUDA_ERROR(err, "Failed to free device array d_cscColPtr");
        err = cudaFree(d_cscRowIdx);
        CUDA_ERROR(err, "Failed to free device array d_cscRowIdx");
        err = cudaFree(d_cscVal);
        CUDA_ERROR(err, "Failed to free device array d_cscVal");
        return -1;
    }
    err = cudaMalloc(&p_buffer, P_bufferSize);
    CUDA_ERROR(err, "Failed to allocate device vector p_buffer");

    // Start computation
    Timer<DEVICE> TM_device;
    TM_device.start();
    cusparseCsr2cscEx2(
        handle,
        m,
        n,
        nnz,
        d_csrVal,
        d_csrRowPtr,
        d_csrColIdx,
        d_cscVal,
        d_cscColPtr,
        d_cscRowIdx,
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        p_buffer
    );
    err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch cusparseCsr2cscEx2 algo 1");
    // Take time
    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ALGO1: ");
    // Get result from device
    err = cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscColPtr from device to host");
    err = cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscRowIdx from device to host");
    err = cudaMemcpy(cscVal, d_cscVal, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscVal from device to host");
    // Cleaner
    cusparseDestroy(handle);
    err = cudaFree(d_csrRowPtr);
    CUDA_ERROR(err, "Failed to free device array d_csrRowPtr");
    err = cudaFree(d_csrColIdx);
    CUDA_ERROR(err, "Failed to free device array d_csrColIdx");
    err = cudaFree(d_csrVal);
    CUDA_ERROR(err, "Failed to free device array d_csrVal");
    err = cudaFree(d_cscColPtr);
    CUDA_ERROR(err, "Failed to free device array d_cscColPtr");
    err = cudaFree(d_cscRowIdx);
    CUDA_ERROR(err, "Failed to free device array d_cscRowIdx");
    err = cudaFree(d_cscVal);
    CUDA_ERROR(err, "Failed to free device array d_cscVal");

    return TM_device.duration(); 
}


float nvidia2(
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
    cudaSetDevice(0);
    // Check if memory is enough
    double reqMem = (nnz * sizeof(int)) * 2 + (nnz * sizeof(double)) * 2 + (m+1) * sizeof(int) + (n+1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();    
    if ( nvidiaFreeMemory < reqMem) {
        return -1;
    }
        
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    int     *d_csrRowPtr;
    int     *d_csrColIdx;
    double  *d_csrVal;
    int     *d_cscColPtr;
    int     *d_cscRowIdx;
    double  *d_cscVal;
    // Set device memory
    err = cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscColPtr");
    err = cudaMalloc(&d_cscRowIdx, nnz   * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscRowIdx");
    err = cudaMalloc(&d_cscVal,    nnz   * sizeof(double));
    CUDA_ERROR(err, "Failed to allocate device vector d_cscVal");
    err = cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrRowPtr");
    err = cudaMalloc(&d_csrColIdx, nnz   * sizeof(int));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrColIdx");
    err = cudaMalloc(&d_csrVal,    nnz   * sizeof(double));
    CUDA_ERROR(err, "Failed to allocate device vector d_csrVal");
    err = cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrRowPtr from host to device");
    err = cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrColIdx from host to device");
    err = cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_ERROR(err, "Failed to copy array csrVal from host to device");
    
    // setup buffersize
    size_t  P_bufferSize = 0;
    char*   p_buffer= NULL;

    cusparseCsr2cscEx2_bufferSize(
        handle,
        m,
        n,
        nnz,
        d_csrVal,
        d_csrRowPtr,
        d_csrColIdx,
        d_cscVal,
        d_cscColPtr,
        d_cscRowIdx,
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG2,
        &P_bufferSize
    );

    reqMem = reqMem + static_cast<double>(P_bufferSize);
    if (nvidiaFreeMemory < reqMem) {
        err = cudaFree(d_csrRowPtr);
        CUDA_ERROR(err, "Failed to free device array d_csrRowPtr");
        err = cudaFree(d_csrColIdx);
        CUDA_ERROR(err, "Failed to free device array d_csrColIdx");
        err = cudaFree(d_csrVal);
        CUDA_ERROR(err, "Failed to free device array d_csrVal");
        err = cudaFree(d_cscColPtr);
        CUDA_ERROR(err, "Failed to free device array d_cscColPtr");
        err = cudaFree(d_cscRowIdx);
        CUDA_ERROR(err, "Failed to free device array d_cscRowIdx");
        err = cudaFree(d_cscVal);
        CUDA_ERROR(err, "Failed to free device array d_cscVal");
        return -1;
    }
    err = cudaMalloc(&p_buffer, P_bufferSize);
    CUDA_ERROR(err, "Failed to allocate device vector p_buffer");
    
    // Start computation
    Timer<DEVICE> TM_device;
    TM_device.start();
    cusparseCsr2cscEx2(
        handle,
        m,
        n,
        nnz,
        d_csrVal,
        d_csrRowPtr,
        d_csrColIdx,
        d_cscVal,
        d_cscColPtr,
        d_cscRowIdx,
        CUDA_R_64F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG2,
        p_buffer
    );
    err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch cusparseCsr2cscEx2 algo 2");
    // Take time
    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ALGO2: ");
    // Copy result from device
    err = cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscColPtr from device to host");
    err = cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscRowIdx from device to host");
    err = cudaMemcpy(cscVal, d_cscVal, nnz * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy array d_cscVal from device to host");
    // Cleaner
    cusparseDestroy(handle);
    err = cudaFree(d_csrRowPtr);
    CUDA_ERROR(err, "Failed to free device array d_csrRowPtr");
    err = cudaFree(d_csrColIdx);
    CUDA_ERROR(err, "Failed to free device array d_csrColIdx");
    err = cudaFree(d_csrVal);
    CUDA_ERROR(err, "Failed to free device array d_csrVal");
    err = cudaFree(d_cscColPtr);
    CUDA_ERROR(err, "Failed to free device array d_cscColPtr");
    err = cudaFree(d_cscRowIdx);
    CUDA_ERROR(err, "Failed to free device array d_cscRowIdx");
    err = cudaFree(d_cscVal);
    CUDA_ERROR(err, "Failed to free device array d_cscVal");

    return TM_device.duration();
}