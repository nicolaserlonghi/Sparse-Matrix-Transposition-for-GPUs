#ifndef _SERIAL_H
#define _SERIAL_H

#include <iostream>

float serial(
    int m,
    int n,
    int nnz,
    int *csrRowPtr,
    int *csrColIdx,
    double *csrVal,
    int *cscColPtr,
    int *cscRowIdx,
    double *cscVal
);

#endif