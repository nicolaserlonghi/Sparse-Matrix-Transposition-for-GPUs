#include <serial.h>

void serial(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal, int *cscColPtr, int *cscRowIdx, double *cscVal) {
    int *curr = new int[n]();
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            cscColPtr[csrColIdx[j] + 1]++;
        }
    }
    
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

    free(curr);
}