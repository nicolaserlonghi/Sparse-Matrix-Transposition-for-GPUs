#ifndef _SCAN_TRANS_ROW_ROW_H
#define _SCAN_TRANS_ROW_ROW_H

#include <iostream>

float scanTransRowRow(
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