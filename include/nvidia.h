#ifndef _NVIDIA_H
#define _NVIDIA_H

#include <cusparse_v2.h>

#define CUDA_ERROR( err, msg ) { \
    if (err != cudaSuccess) {\
        printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
        exit( EXIT_FAILURE );\
    }\
}

float nvidia(
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

float nvidia2(
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