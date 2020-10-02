#ifndef _SCAN_TRANS_H
#define _SCAN_TRANS_H

#include <iostream>

float scanTrans(
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

bool checkIsMemoryNotEnough(int n, int m, int nnz);

void getOccupancyMaxPotentialBlockSize(
    int biggest,
    int *gridSizeHistogram,
    int *gridSizeVerticalScan,
    int *gridSizeWriteBack,
    int *blockSizeHistogram,
    int *blockSizeVerticalScan,
    int *blockSizeWriteBack
);

#endif