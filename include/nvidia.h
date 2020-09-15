#include <cusparse_v2.h>

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
);