#include <nvidia.h>
#include <utilities.h>
#include <Timer.cuh>

using namespace timer;

float nvidia(
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
    cudaSetDevice(0);

    double reqMem = (nnz * sizeof(int)) * 2 + (nnz * sizeof(double)) * 2 + (m+1) * sizeof(int) + (n+1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();    
    if ( nvidiaFreeMemory < reqMem) {
        std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
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

    // Matrix csr
    cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMalloc(&d_csrColIdx, nnz   * sizeof(int));
    cudaMalloc(&d_csrVal,    nnz   * sizeof(double));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz   * sizeof(double), cudaMemcpyHostToDevice);

    // Matrix csc     
    cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int));
    cudaMalloc(&d_cscRowIdx, nnz   * sizeof(int));
    cudaMalloc(&d_cscVal,    nnz   * sizeof(double));
    
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
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
    if ( nvidiaFreeMemory < reqMem) {
        cudaFree(d_csrRowPtr);
        cudaFree(d_csrColIdx);
        cudaFree(d_csrVal);
        cudaFree(d_cscColPtr);
        cudaFree(d_cscRowIdx);
        cudaFree(d_cscVal);
        return -1;
    }

    cudaMalloc(&p_buffer, P_bufferSize);
    
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

    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ALGO1: ");

  
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // step 6: free resources

    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);

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

    Timer<DEVICE> TM_device;
    cudaSetDevice(0);

    double reqMem = (nnz * sizeof(int)) * 2 + (nnz * sizeof(double)) * 2 + (m+1) * sizeof(int) + (n+1) * sizeof(int);
    double nvidiaFreeMemory = getSizeOfNvidiaFreeMemory();    
    if ( nvidiaFreeMemory < reqMem) {
        std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
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

    // Matrix csr
    cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMalloc(&d_csrColIdx, nnz   * sizeof(int));
    cudaMalloc(&d_csrVal,    nnz   * sizeof(double));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz   * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz   * sizeof(double), cudaMemcpyHostToDevice);

    // Matrix csc     
    cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int));
    cudaMalloc(&d_cscRowIdx, nnz   * sizeof(int));
    cudaMalloc(&d_cscVal,    nnz   * sizeof(double));
    
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
    std::cout << std::setprecision(0) << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
    if ( nvidiaFreeMemory < reqMem) {
        cudaFree(d_csrRowPtr);
        cudaFree(d_csrColIdx);
        cudaFree(d_csrVal);
        cudaFree(d_cscColPtr);
        cudaFree(d_cscRowIdx);
        cudaFree(d_cscVal);
        return -1;
    }
    cudaMalloc(&p_buffer, P_bufferSize);
    
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

    TM_device.stop();
    TM_device.print("GPU Sparse Matrix Transpostion ALGO2: ");

  
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // step 6: free resources

    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);

    return TM_device.duration(); 

}