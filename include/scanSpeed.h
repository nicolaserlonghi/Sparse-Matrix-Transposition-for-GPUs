#ifndef _SCAN_SPEED_H
#define _SCAN_SPEED_H

#include <iostream>

float scanSpeed(
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