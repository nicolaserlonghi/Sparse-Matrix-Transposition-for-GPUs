#include <nvidia.h>
#include <utilities.h>
#include <Timer.cuh>
#include <CheckError.cuh>

using namespace timer;

using namespace std;


void cuda_sptrans(
    int         m,
    int         n,
    int         nnz,
    int        *csrRowPtr,
    int        *csrColIdx,
    double     *csrVal,
    int        *cscColPtr,
    int        *cscRowIdx,
    double     *cscVal
) {
    Timer<DEVICE> TM_device;
    cudaSetDevice(0);
    
    cusparseHandle_t handle = NULL;

    cusparseCreate(&handle);

    int *d_csrRowPtr;
    int *d_csrColIdx;
    double *d_csrVal;
   
    int *d_cscColPtr;
    int *d_cscRowIdx;
    double *d_cscVal;

    // Qui inizia il calcolo del tempo di copia dei dati

    // TODO: verificare cosa fa il mallocManaged rispetto a Malloc
    // TODO: verificare perché serve proprio memset

    // Matrix csr
    SAFE_CALL( cudaMallocManaged((void **)&d_csrRowPtr, (m+1) * sizeof(int)) );
    SAFE_CALL( cudaMallocManaged((void **)&d_csrColIdx, nnz   * sizeof(int)) );
    SAFE_CALL( cudaMallocManaged((void **)&d_csrVal,    nnz   * sizeof(double)) );

    SAFE_CALL( cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),   cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(d_csrColIdx, csrColIdx, nnz  * sizeof(int),    cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(d_csrVal,    csrVal,    nnz  * sizeof(double), cudaMemcpyHostToDevice) );

    // Matrix csc     
    SAFE_CALL( cudaMallocManaged((void **)&d_cscColPtr, (n+1) * sizeof(int)) );
    SAFE_CALL( cudaMallocManaged((void **)&d_cscRowIdx, nnz   * sizeof(int)) );
    SAFE_CALL( cudaMallocManaged((void **)&d_cscVal,    nnz   * sizeof(double)) );

    SAFE_CALL( cudaMemset(d_cscColPtr, 0, (n+1) * sizeof(int)) );
    SAFE_CALL( cudaMemset(d_cscRowIdx, 0, nnz   * sizeof(int)) );
    SAFE_CALL( cudaMemset(d_cscVal,    0, nnz   * sizeof(double)) );

    // Qui finisce il tempo per la copia dei dati
    
    // setup buffersize

    TM_device.start();

    // Qui andrebbero i DimGrid e DimBlock

    size_t P_bufferSize = 0;

    char* p_buffer= NULL;

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
                            CUDA_C_32F,
                            CUSPARSE_ACTION_NUMERIC,
                            CUSPARSE_INDEX_BASE_ZERO,
                            CUSPARSE_CSR2CSC_ALG1,
                            &P_bufferSize);

    printf("P_bufferSize  = %lld \n", (long long)P_bufferSize);

    if (NULL != p_buffer) { 
        SAFE_CALL( cudaFree(p_buffer) );
    }

    SAFE_CALL( cudaMalloc((void**)&p_buffer, P_bufferSize) );
    
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
		        CUDA_C_32F,
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG1,
                p_buffer);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("GPU Sparse Matrix Transpostion: ");

  
    SAFE_CALL( cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),  cudaMemcpyDeviceToHost) );
    SAFE_CALL( cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int),    cudaMemcpyDeviceToHost) );
    SAFE_CALL( cudaMemcpy(cscVal,    d_cscVal,    nnz * sizeof(double), cudaMemcpyDeviceToHost) );

    // step 6: free resources

    cusparseDestroy(handle);
    SAFE_CALL( cudaFree(d_csrRowPtr) );
    SAFE_CALL( cudaFree(d_csrColIdx) );
    SAFE_CALL( cudaFree(d_csrVal) );
    SAFE_CALL( cudaFree(d_cscColPtr) );
    SAFE_CALL( cudaFree(d_cscRowIdx) );
    SAFE_CALL( cudaFree(d_cscVal) );

}