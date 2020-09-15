#include <cusparse_v2.h>

using namespace std;

int cuda_sptrans(
    const int         m,
    const int         n,
    const int         nnz,
    const int        *csrRowPtr,
    const int        *csrColIdx,
    const double     *csrVal,
    int              *cscRowIdx,
    int              *cscColPtr,
    double           *cscVal
) {
    cudaSetDevice(0);
    
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;

    status = cusparseCreate(&handle);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    status = cusparseSetStream(handle, stream);

    struct timeval t10, t11;
    double time_memory_copy=0;
    gettimeofday(&t10,NULL);
    
    int *d_csrRowPtr;
    int *d_csrColIdx;
    double *d_csrVal;
   
    int *d_cscColPtr;
    int *d_cscRowIdx;
    double *d_cscVal;

    // Matrix csr
    cudaMallocManaged((void **)&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMallocManaged((void **)&d_csrColIdx, nnz  * sizeof(int));
    cudaMallocManaged((void **)&d_csrVal,    nnz  * sizeof(double));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz  * sizeof(double),   cudaMemcpyHostToDevice);

    // Matrix csc     
    cudaMallocManaged((void **)&d_cscColPtr, (n+1) * sizeof(int));
    cudaMallocManaged((void **)&d_cscRowIdx, nnz  * sizeof(int));
    cudaMallocManaged((void **)&d_cscVal,    nnz  * sizeof(double));

    cudaMemset(d_cscColPtr, 0, (n+1) * sizeof(int));
    cudaMemset(d_cscRowIdx, 0, nnz  * sizeof(int));
    cudaMemset(d_cscVal,    0,    nnz  * sizeof(double));

    gettimeofday(&t11, NULL);
    time_memory_copy = (t11.tv_sec - t10.tv_sec) * 1000.0 + (t11.tv_usec - t10.tv_usec) / 1000.0;

    printf("cuSparse memory copy for single gpu used %4.2f ms,\n",time_memory_copy);

    struct timeval t3, t4;
    double time_cuda_trans= 0;
    gettimeofday(&t3, NULL); 
    
    // setup buffersize

    size_t P_bufferSize = 0;

    char* p_buffer= NULL;

    status = cusparseCsr2cscEx2_bufferSize(handle,
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
        cudaFree(p_buffer); 
    }

    cudaStat1 = cudaMalloc((void**)&p_buffer, P_bufferSize);
    
    status = cusparseCsr2cscEx2(
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
    
    if (CUSPARSE_STATUS_INTERNAL_ERROR == status) printf("CUSPARSE_STATUS_INTERNAL_ERROR\n");
    cudaStat1 = cudaDeviceSynchronize();

    gettimeofday(&t4, NULL);
    time_cuda_trans = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    
    cout << "cuSparse trans used " << time_cuda_trans << " ms" << endl;
   
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz  * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal, d_cscVal,   nnz  * sizeof(double),   cudaMemcpyDeviceToHost);

    // step 6: free resources

    cusparseDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);
    
    return 0;
}