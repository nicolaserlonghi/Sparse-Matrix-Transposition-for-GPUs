#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <iostream>


void printArray(int m, int *array);

char* detectFile(int argc, char *argv);

void readMatrix(
    char *filename,
    int &m,
    int &n,
    int &nnz,
    int *&csrRowPtr,
    int *&csrColIdx,
    double *&csrVal
);

void clearTheBuffers(
    int n,
    int nnz,
    int *cscRowIdx,
    double *cscVal,
    int *cscColPtr
);

float performTransposition(
    float (*f)(int, int, int, int*, int*, double*, int*, int*, double*),
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

double getSizeOfNvidiaFreeMemory();

#endif