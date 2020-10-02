#include <serial.h>
#include <utilities.h>
#include <Timer.cuh>

using namespace timer;

float serial(
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

    Timer<HOST> TM_host;

    TM_host.start();
    int *curr = new int[n]();
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            cscColPtr[csrColIdx[j] + 1]++;
        }
    }
    
    // prefix sum
    for(int i = 1; i < n + 1; i++) {
        cscColPtr[i] += cscColPtr[i - 1];
    }

    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int loc = cscColPtr[csrColIdx[j]] + curr[csrColIdx[j]]++;
            cscRowIdx[loc] = i;
            cscVal[loc] = csrVal[j];
        }
    }

    TM_host.stop();
    TM_host.print("Serial Sparse Matrix Transpostion: ");

    free(curr);

    return TM_host.duration(); 
}