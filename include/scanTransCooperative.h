#ifndef _SCAN_TRANS_COOPERATIVE_H
#define _SCAN_TRANS_COOPERATIVE_H

#include <iostream>

float scanTransCooperative(
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal
);

#endif