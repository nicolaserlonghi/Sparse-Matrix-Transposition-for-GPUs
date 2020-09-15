#include <utilities.h>
#include <scanTrans.h>

void scanTrans(
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
    int nthreads = 1;
    int nthread = 0;
    int tid = 0;
    int start = 0;
    int *intra = new int[nnz]();
    int *inter = new int[(nthreads + 1) * n]();
    int *csrRowIdx = new int[nnz]();
    int len = nnz;
    for(int i = 0; i < m; i++) {
      for(int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
        csrRowIdx[j] = i;
      }
    }

    for(int i = 0; i < len; i++) {
      intra[start + i] = inter[(tid + 1) * n + csrColIdx[start + i]]++;
    }

    for(int i = 0; i < n; i++) {
      for(int j = 1; j < nthread + 1; j++) {
        inter[i + n * j] += inter[i + n * (j - 1)];
     }
    }

    for(int i = 0; i < n; i++) {
      for(int j = 0; j < nthreads; j++) {
        inter[i + n * j] += inter[i + n * (j - 1)];
       }
    }

    for(int i = 0; i < n; i++) {
      cscColPtr[i + 1] = inter[n * nthread + i];
    }

    // prefixSum()
    for(int i = 1; i < n + 1; i++) {
        cscColPtr[i] += cscColPtr[i - 1];
    }

    for(int i = 0; i < len; i++) {
      int loc = cscColPtr[csrColIdx[start + i]] + inter[tid * n + csrColIdx[start + i]] + intra[start + i];
      csrRowIdx[loc] = csrRowIdx[start + i];
      cscVal[loc] = cscVal[start + i];
    }

    printArray(nnz,cscVal);
   
    free(intra);
    free(inter);
    free(csrRowIdx);
}