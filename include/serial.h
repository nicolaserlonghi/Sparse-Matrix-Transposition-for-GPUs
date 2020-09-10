#ifndef _SERIAL_H
#define _SERIAL_H

template<typename iT, typename vT>
void serial(int m, int n, int nnz, iT *csrRowPtr, iT *csrColIdx, vT *csrVal, iT *cscColPtr, iT *cscRowIdx, vT *cscVal) {

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

#endif