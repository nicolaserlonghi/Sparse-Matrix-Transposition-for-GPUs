#include <nvidia.h>
#include <utilities.h>
#include <Timer.cuh>
#include <CheckError.cuh>

using namespace timer;

int nvidia(
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
    std::cout << "reqMem: " << reqMem << " free memory: " << nvidiaFreeMemory << std::endl;
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

    // Matrix csr
    SAFE_CALL(cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrColIdx, nnz   * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrVal,    nnz   * sizeof(double)));

    SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrColIdx, csrColIdx, nnz   * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrVal,    csrVal,    nnz   * sizeof(double), cudaMemcpyHostToDevice));

    // Matrix csc     
    SAFE_CALL(cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscRowIdx, nnz   * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscVal,    nnz   * sizeof(double)));
    
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

    SAFE_CALL(cudaMalloc(&p_buffer, P_bufferSize));
    
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
    CHECK_CUDA_ERROR
    TM_device.print("GPU Sparse Matrix Transpostion ALGO1: ");

  
    SAFE_CALL(cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost));

    // step 6: free resources

    cusparseDestroy(handle);
    SAFE_CALL(cudaFree(d_csrRowPtr));
    SAFE_CALL(cudaFree(d_csrColIdx));
    SAFE_CALL(cudaFree(d_csrVal));
    SAFE_CALL(cudaFree(d_cscColPtr));
    SAFE_CALL(cudaFree(d_cscRowIdx));
    SAFE_CALL(cudaFree(d_cscVal));

    return 0;

}

int nvidia2(
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
    if ( getSizeOfNvidiaFreeMemory() < reqMem) {
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
    SAFE_CALL(cudaMalloc(&d_csrRowPtr, (m+1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrColIdx, nnz   * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_csrVal,    nnz   * sizeof(double)));

    SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrColIdx, csrColIdx, nnz   * sizeof(int),    cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_csrVal,    csrVal,    nnz   * sizeof(double), cudaMemcpyHostToDevice));

    // Matrix csc     
    SAFE_CALL(cudaMalloc(&d_cscColPtr, (n+1) * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscRowIdx, nnz   * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_cscVal,    nnz   * sizeof(double)));
    
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

    SAFE_CALL(cudaMalloc(&p_buffer, P_bufferSize));
    
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
    CHECK_CUDA_ERROR
    TM_device.print("GPU Sparse Matrix Transpostion ALGO2: ");

  
    SAFE_CALL(cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost));

    // step 6: free resources

    cusparseDestroy(handle);
    SAFE_CALL(cudaFree(d_csrRowPtr));
    SAFE_CALL(cudaFree(d_csrColIdx));
    SAFE_CALL(cudaFree(d_csrVal));
    SAFE_CALL(cudaFree(d_cscColPtr));
    SAFE_CALL(cudaFree(d_cscRowIdx));
    SAFE_CALL(cudaFree(d_cscVal));

    return 0;

}