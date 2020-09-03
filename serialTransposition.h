#ifndef _SERIALTRANSPOSITION_H
#define _SERIALTRANSPOSITION_H

template<typename iT, typename vT>
void serialTransposition(
    int  m,
    int  n,
    int  nnz,
    iT  *csrRowPtr,
    iT  *csrColIdx,
    vT  *csrVal,    
    iT  *cscColPtr,
    iT  *cscRowIdx,
    vT  *cscVal
) {
    // Construct an array of size n to record current available position in each column
    int *curr = new int[n]();
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            cscColPtr[csrColIdx[j] + 1]++;
        }
    }
    
    // Prefix Sum
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